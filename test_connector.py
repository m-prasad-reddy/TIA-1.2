import asyncio
from database.connector import DatabaseConnection

async def test_bikestores():
    dc = DatabaseConnection('bikestores')
    print("bikestores credentials:", dc.display_name, dc.db_name, dc.sql_credentials, dc.s3_credentials)
    await dc.connect()
    print("bikestores health:", await dc.check_health())
    await dc.close()

async def test_bikestores_csv():
    dc = DatabaseConnection('bikestores-csv')
    print("bikestores-csv credentials:", dc.display_name, dc.db_name, dc.sql_credentials, dc.s3_credentials)
    await dc.connect()
    print("bikestores-csv health:", await dc.check_health())
    await dc.close()

async def test_invalid():
    dc = DatabaseConnection('invalid')
    print("invalid config:", dc.display_name, dc.db_name)
    await dc.close()

async def main():
    print("Testing bikestores:")
    await test_bikestores()
    print("\nTesting bikestores-csv:")
    await test_bikestores_csv()
    print("\nTesting invalid config:")
    await test_invalid()

if __name__ == "__main__":
    asyncio.run(main())