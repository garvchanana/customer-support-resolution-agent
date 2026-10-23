
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customers.id"), nullable=False, index=True
    )
    product: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    paid_amount_minor: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    delivered_at: Mapped[date | None] = mapped_column(Date, nullable=True)


class ReturnRecord(Base):
    __tablename__ = "returns"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    order_id: Mapped[str] = mapped_column(
        ForeignKey("orders.id"), unique=True, nullable=False
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    inspection_status: Mapped[str] = mapped_column(
        String(30), nullable=False
    )


class Payment(Base):
    __tablename__ = "payments"

    order_id: Mapped[str] = mapped_column(
        ForeignKey("orders.id"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    refunded_amount_minor: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customers.id"), nullable=False, index=True
    )
    order_id: Mapped[str | None] = mapped_column(
        ForeignKey("orders.id"), nullable=True
    )
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    message: Mapped[str] = mapped_column(String(2000), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
