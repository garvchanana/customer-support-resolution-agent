
from pprint import pprint

from app.agent.workflow import support_graph


result = support_graph.invoke({
    "customer_id": "cus_001",
    "order_id": "ord_1001",
    "message": "I would like a refund for my headphones.",
})

pprint(result)
