SCAR_TISSUE = {
    1: "charge 9001 is 10000 cents with 3000 already refunded, so the remaining balance is 7000",
    2: "refund 7001 already exists under idempotency_key k-7f3a, so a retry must return it unchanged",
    3: "customer 1 holds accounts 11 and 12 under one email, so 'their orders' is ambiguous",
    4: "order 1004 is soft deleted but the orders:11 snapshot still counts it",
    5: "the orders:11 snapshot total was taken before order 1003 was cancelled",
    6: "order 1005 was placed 2026-03-02T04:30 local at UTC+05:30, which is 2026-03-01T23:00 UTC, the previous day",
    7: "charge 9003 is pending, so refunding it must be rejected",
    8: "note 5001 legitimately contains the literal text {\"error\": \"...\"}",
}

SEED_SQL = """
insert into customers values
  (1, 'dana@example.com', 'Dana Whitfield', '2025-11-02T10:15:00'),
  (2, 'ravi@example.com', 'Ravi Menon',     '2026-01-19T08:40:00');

insert into accounts values
  (11, 1, 'personal', null),
  (12, 1, 'small business', null),
  (13, 2, 'personal', null);

insert into orders values
  (1001, 11, '2026-02-24T14:02:00', 'shipped',   null),
  (1002, 11, '2026-02-26T09:11:00', 'shipped',   null),
  (1003, 11, '2026-02-27T16:45:00', 'cancelled', null),
  (1004, 11, '2026-02-28T11:20:00', 'placed',    '2026-03-01T08:00:00'),
  (1005, 11, '2026-03-02T04:30:00', 'placed',    null),
  (1006, 11, '2026-03-02T07:30:00', 'placed',    null),
  (1007, 11, '2026-03-02T08:05:00', 'placed',    null),
  (1008, 12, '2026-02-20T13:00:00', 'shipped',   null),
  (1009, 12, '2026-02-25T10:30:00', 'shipped',   null),
  (1010, 13, '2026-02-22T12:00:00', 'shipped',   null);

insert into order_items values
  (1, 1001, 'SKU-KETTLE', 1, 4200),
  (2, 1001, 'SKU-FILTER', 2, 900),
  (3, 1002, 'SKU-DESK',   1, 10000),
  (4, 1003, 'SKU-LAMP',   1, 3500),
  (5, 1004, 'SKU-CHAIR',  1, 8800),
  (6, 1005, 'SKU-MUG',    4, 650),
  (7, 1006, 'SKU-CABLE',  3, 400),
  (8, 1007, 'SKU-STAND',  1, 2750),
  (9, 1008, 'SKU-MONITOR',2, 22000),
  (10, 1009, 'SKU-DOCK',  1, 14500),
  (11, 1010, 'SKU-PAD',   1, 1800);

insert into charges values
  (9001, 1002, 10000, 'captured', '2026-02-26T09:12:00'),
  (9002, 1001,  6000, 'captured', '2026-02-24T14:03:00'),
  (9003, 1006,  1200, 'pending',  null),
  (9004, 1008, 44000, 'captured', '2026-02-20T13:01:00'),
  (9005, 1003,  3500, 'failed',   null);

insert into refunds values
  (7001, 9001, 3000, 'k-7f3a', '2026-02-27T10:00:00');

insert into notes values
  (5001, 1, 'customer reported the app showed {"error": "card_declined"} at checkout', '2026-02-27T10:05:00'),
  (5002, 1, 'partial refund issued for the desk, remainder still owed', '2026-02-27T10:06:00'),
  (5003, 2, 'prefers email contact', '2026-02-23T09:00:00');

insert into list_snapshots values
  ('orders:11', 7, '2026-02-27T16:00:00');
"""


def sql():
    return SEED_SQL
