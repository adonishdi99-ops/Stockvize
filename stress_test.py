"""
Rougher stress test for agent_core_groq.py. Run after the basic test passes.

Usage: python3 stress_test.py
"""

from agent_core_groq import handle_message

stress_messages = [
    ("customer", "abeg i wan buy 3 bag rise n 2 beans deliver ikeja"),
    ("vendor", "d garri don finish, remove am from stock make e be 0"),
    ("customer", "please send two bags of rice to my house"),
    ("customer", "1 rice, 5 cups beans, and 2 garri"),
    ("customer", "how much is rice? also send me 1 bag"),
    ("customer", "I want some rice delivered to Iwo road"),
    ("vendor", "sorry I said add 20 garri, I meant 10, fix it"),
    ("vendor", "that last order don deliver"),
    ("customer", "has my order been delivered yet"),
    ("customer", "hello are you the shop that sells generators"),
]

if __name__ == "__main__":
    for role, msg in stress_messages:
        print(f"\n> [{role}] {msg}")
        result = handle_message(role, msg)
        print(result["debug_log"])
        print(f"  [to customer] {result['customer_reply']}")