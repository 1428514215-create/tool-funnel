"""Shared fixtures for the tool-funnel test suite."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolfunnel import (  # noqa: E402
    DomainSpec,
    Funnel,
    ParamSpec,
    StageSpec,
    ToolSpec,
)


@pytest.fixture
def small_tools():
    return [
        ToolSpec(
            name="order.search", domain="order", stage="locate",
            description="按关键字搜索订单",
            keywords=["订单", "搜索", "order"],
            params={"keyword": ParamSpec("keyword", required=True)},
            produces=["order_id"],
            next_hint=["order.get"],
        ),
        ToolSpec(
            name="order.get", domain="order", stage="inspect",
            description="按订单号查看订单详情",
            keywords=["订单", "详情", "order"],
            params={"order_id": ParamSpec("order_id", required=True, aliases=["id"])},
            requires=["order_id"],
            produces=["order_status"],
            next_hint=["shipment.track"],
        ),
        ToolSpec(
            name="shipment.track", domain="logistics", stage="track",
            description="跟踪物流轨迹",
            keywords=["物流", "轨迹", "track"],
            params={"order_id": ParamSpec("order_id", required=True)},
            requires=["order_id"],
            bridges=["order"],
        ),
        ToolSpec(
            name="refund.create", domain="after_sales", stage="refund",
            description="创建退款单",
            keywords=["退款", "售后", "refund"],
            params={"order_id": ParamSpec("order_id", required=True),
                    "amount": ParamSpec("amount", default="full")},
            produces=["refund_id"],
        ),
    ]


@pytest.fixture
def small_domains():
    return [
        DomainSpec("order", triggers=("订单", "order"), bridges=("logistics",)),
        DomainSpec("logistics", triggers=("物流", "快递"), bridges=("order",)),
        DomainSpec("after_sales", triggers=("退款", "售后"), bridges=("order",)),
    ]


@pytest.fixture
def small_stages():
    return [
        StageSpec("locate", "order", triggers=("查", "搜索"), order=0),
        StageSpec("inspect", "order", triggers=("详情", "状态"), order=1),
        StageSpec("track", "logistics", triggers=("物流", "轨迹"), order=0),
        StageSpec("refund", "after_sales", triggers=("退款", "退"), order=2),
    ]


@pytest.fixture
def funnel(small_tools, small_domains, small_stages):
    return Funnel.from_tools(small_tools, domains=small_domains, stages=small_stages)
