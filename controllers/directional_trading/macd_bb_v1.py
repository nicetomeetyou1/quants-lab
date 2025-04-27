from typing import List

from decimal import Decimal

import pandas_ta as ta  # noqa: F401
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.core.data_type.common import PositionAction, PriceType, TradeType
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig

class MACDBBV1ControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "macd_bb_v1"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",
            "prompt_on_new": True})
    candles_trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",
            "prompt_on_new": True})
    interval: str = Field(
        default="3m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True})
    bb_length: int = Field(
        default=100,
        json_schema_extra={"prompt": "Enter the Bollinger Bands length: ", "prompt_on_new": True})
    bb_std: float = Field(default=2.0)
    bb_long_threshold: float = Field(default=0.0)
    bb_short_threshold: float = Field(default=1.0)
    natr_length: float = Field(
        default=200,
        json_schema_extra={"prompt": "Enter the NATR length: ", "prompt_on_new": True}
    )
    position_sizing_method: str = Field(
        default="fix",
        json_schema_extra={"Prompt": "Enter position sizing method fix or volatility", "prompt_on_new": True}
    )
    distribution_method: str = Field(
        default="fibonacci",
        json_schema_extra={"Prompt": "Enter distribution method: equal, fibonacci, inverted_triangle",
                           "prompt_on_new": True}
    )
    step_percentage: Decimal = Field(
        default=Decimal("0.0005"), gt=0,
        json_schema_extra={
            "prompt": "Enter the step percentage (as a decimal, e.g., 0.0005 for 0.05%): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    macd_fast: int = Field(
        default=21,
        json_schema_extra={"prompt": "Enter the MACD fast period: ", "prompt_on_new": True})
    macd_slow: int = Field(
        default=42,
        json_schema_extra={"prompt": "Enter the MACD slow period: ", "prompt_on_new": True})
    macd_signal: int = Field(
        default=9,
        json_schema_extra={"prompt": "Enter the MACD signal period: ", "prompt_on_new": True})

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class MACDBBV1Controller(DirectionalTradingControllerBase):

    def __init__(self, config: MACDBBV1ControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(config.macd_slow, config.macd_fast, config.macd_signal, config.bb_length) + 20
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                      trading_pair=self.config.candles_trading_pair,
                                                      interval=self.config.interval,
                                                      max_records=self.max_records)
        # Add indicators
        df.ta.bbands(length=self.config.bb_length, std=self.config.bb_std, append=True)
        df.ta.macd(fast=self.config.macd_fast, slow=self.config.macd_slow, signal=self.config.macd_signal, append=True)
        
        # Get volatility factor to for dynamic tp or sl
        df["volatility_factor"] = ta.natr(high=df["high"],
            low=df["low"],
            close=df["close"],
            length=self.config.natr_length,
            append=True
        )

        bbp = df[f"BBP_{self.config.bb_length}_{self.config.bb_std}"]
        macdh = df[f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        macd = df[f"MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]

        
        # Generate signal
        long_condition = (bbp < self.config.bb_long_threshold) & (macdh > 0) & (macd < 0)
        short_condition = (bbp > self.config.bb_short_threshold) & (macdh < 0) & (macd > 0)

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["volatility_factor"] = df["volatility_factor"].iloc[-1]
        self.processed_data["features"] = df

    async def get_position_hold(self):
        pnl = sum([pos.global_pnl_quote for pos in self.positions_held])
        return pnl
    
    def get_executor_config(self, trade_type: TradeType, price: Decimal, amount: Decimal):
        # Return triple barrier config with adjust stop loss and take profit based on NATR
        position = PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=trade_type,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config.new_instance_with_adjusted_volatility(self.processed_data["volatility_factor"]),
            leverage=self.config.leverage,
        )
        # print(position.dict())
        return position
    
    def create_actions_proposal(self) -> List[ExecutorAction]:
        """
        Create actions based on the provided executor handler report.
        """
        create_actions = []
        signal = self.processed_data["signal"]
        if signal != 0 and self.can_create_executor(signal):
            price = self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair,
                                                                PriceType.MidPrice)
            # Default implementation distribute the total amount equally among the executors

            amount = self.config.total_amount_quote / price / self.config.max_executors_per_side
            order_count = Decimal(self.config.max_executors_per_side)
            order_sizes: List[Decimal] = []
            base_order_size = 0

            print(self.processed_data["volatility_factor"])

            if self.config.position_sizing_method == "volatility":
                base_order_size = Decimal(amount) * Decimal(self.processed_data["volatility_factor"])
            elif self.config.position_sizing_method == "fix":
                base_order_size = Decimal(amount)
            
            if self.config.distribution_method == "equal":
                order_sizes = [base_order_size for _ in range(order_count)]
            elif self.config.distribution_method == "fibonacci":
                fib_seq = self.fibonacci(self.config.max_executors_per_side)
                factor = base_order_size / max(fib_seq)
                order_sizes = [w * factor for w in fib_seq]
                print(fib_seq)
            elif self.config.distribution_method == "inverted_triangle":
                weights = self.inverted_triangle(self.config.max_executors_per_side)
                print(weights)
                factor = base_order_size / max(weights)
                order_sizes = [w * factor for w in weights]

            trade_type = TradeType.BUY if signal > 0 else TradeType.SELL
            print(order_sizes, amount, self.config.distribution_method, self.config.position_sizing_method)
            for i in range(len(order_sizes)):
                price_order = price * (1 + self.config.step_percentage * i) if trade_type == TradeType.SELL else price * (1 - self.config.step_percentage * i)
                create_actions.append(CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=self.get_executor_config(trade_type, price=price_order, amount=order_sizes[i])))
                
        return create_actions
    
    def fibonacci(self, max_executors_per_side):
        start_fib = [1, 1]
        if max_executors_per_side == 2:
            return start_fib
        
        for i in range(2, max_executors_per_side):
            start_fib.append(start_fib[i - 1] + start_fib[i - 2])
        return start_fib

    def inverted_triangle(self, max_executors_per_side):
        half = max_executors_per_side // 2
        if max_executors_per_side % 2 == 0:
            ascending = list(range(1, half + 1))
            descending = ascending[::-1]
            return ascending + descending
        else:
            ascending = list(range(1, half + 2))  # tengah lebih besar
            descending = ascending[:-1][::-1]
            return ascending + descending


    def stop_action_proposal(self):
        """
        Stop position based on the current signal.
        """
        stop_actions = []

        signal = self.processed_data["signal"]
        if signal == 1:
            # Stop active short position
            stop_actions.extend(self.filter_executors(
                executors=self.executors_info,
                filter_func=lambda x: x.is_active and (x.side == TradeType.SELL)))

        elif signal == -1:
            # Stop active long position
            stop_actions.extend(self.filter_executors(
                executors=self.executors_info,
                filter_func=lambda x: x.is_active and (x.side == TradeType.BUY)))

        return stop_actions
    
