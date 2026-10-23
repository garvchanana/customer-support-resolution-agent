
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class OlistCustomer(Base):
    __tablename__ = "olist_customers"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_unique_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    customer_zip_code_prefix: Mapped[int | None] = mapped_column(Integer)
    customer_city: Mapped[str | None] = mapped_column(String(128))
    customer_state: Mapped[str | None] = mapped_column(String(8))


class OlistProduct(Base):
    __tablename__ = "olist_products"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_category_name: Mapped[str | None] = mapped_column(
        String(128), index=True
    )
    product_name_lenght: Mapped[int | None] = mapped_column(Integer)
    product_description_lenght: Mapped[int | None] = mapped_column(Integer)
    product_photos_qty: Mapped[int | None] = mapped_column(Integer)
    product_weight_g: Mapped[int | None] = mapped_column(Integer)
    product_length_cm: Mapped[int | None] = mapped_column(Integer)
    product_height_cm: Mapped[int | None] = mapped_column(Integer)
    product_width_cm: Mapped[int | None] = mapped_column(Integer)


class OlistSeller(Base):
    __tablename__ = "olist_sellers"

    seller_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    seller_zip_code_prefix: Mapped[int | None] = mapped_column(Integer)
    seller_city: Mapped[str | None] = mapped_column(String(128))
    seller_state: Mapped[str | None] = mapped_column(String(8))


class OlistOrder(Base):
    __tablename__ = "olist_orders"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        ForeignKey("olist_customers.customer_id"), nullable=False, index=True
    )
    order_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    order_purchase_timestamp: Mapped[DateTime | None] = mapped_column(DateTime)
    order_approved_at: Mapped[DateTime | None] = mapped_column(DateTime)
    order_delivered_carrier_date: Mapped[DateTime | None] = mapped_column(DateTime)
    order_delivered_customer_date: Mapped[DateTime | None] = mapped_column(DateTime)
    order_estimated_delivery_date: Mapped[DateTime | None] = mapped_column(DateTime)


class OlistOrderItem(Base):
    __tablename__ = "olist_order_items"

    order_id: Mapped[str] = mapped_column(
        ForeignKey("olist_orders.order_id"), primary_key=True
    )
    order_item_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[str] = mapped_column(
        ForeignKey("olist_products.product_id"), nullable=False, index=True
    )
    seller_id: Mapped[str] = mapped_column(
        ForeignKey("olist_sellers.seller_id"), nullable=False, index=True
    )
    shipping_limit_date: Mapped[DateTime | None] = mapped_column(DateTime)
    price: Mapped[object] = mapped_column(Numeric(12, 2), nullable=False)
    freight_value: Mapped[object] = mapped_column(Numeric(12, 2), nullable=False)


class OlistOrderPayment(Base):
    __tablename__ = "olist_order_payments"

    order_id: Mapped[str] = mapped_column(
        ForeignKey("olist_orders.order_id"), primary_key=True
    )
    payment_sequential: Mapped[int] = mapped_column(Integer, primary_key=True)
    payment_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payment_installments: Mapped[int] = mapped_column(Integer, nullable=False)
    payment_value: Mapped[object] = mapped_column(Numeric(12, 2), nullable=False)


class OlistOrderReview(Base):
    # Use a generated primary key because source review_id values
    # are not guaranteed to be unique in the prepared dataset.
    __tablename__ = "olist_order_reviews"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    review_id: Mapped[str | None] = mapped_column(String(64), index=True)
    order_id: Mapped[str] = mapped_column(
        ForeignKey("olist_orders.order_id"), nullable=False, index=True
    )
    review_score: Mapped[int | None] = mapped_column(Integer)
    review_comment_title: Mapped[str | None] = mapped_column(String(512))
    review_comment_message: Mapped[str | None] = mapped_column(Text)
    review_creation_date: Mapped[DateTime | None] = mapped_column(DateTime)
    review_answer_timestamp: Mapped[DateTime | None] = mapped_column(DateTime)


class OlistProductCategoryTranslation(Base):
    __tablename__ = "olist_product_category_translations"

    product_category_name: Mapped[str] = mapped_column(
        String(128), primary_key=True
    )
    product_category_name_english: Mapped[str] = mapped_column(
        String(128), nullable=False
    )


class BitextMessage(Base):
    __tablename__ = "bitext_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    flags: Mapped[str] = mapped_column(Text, nullable=False)
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    intent: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    response: Mapped[str] = mapped_column(Text, nullable=False)
