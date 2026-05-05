from db import create_tables, get_stats

create_tables()
print("Database connected and tables created!")

stats = get_stats()
print(f"Current applications in DB: {stats['total']}")