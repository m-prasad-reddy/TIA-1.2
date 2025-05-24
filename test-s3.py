import asyncio
import aiobotocore.session

async def test_s3():
    session = aiobotocore.session.get_session()
    async with session.create_client(
        's3',
        aws_access_key_id='accesskey',
        aws_secret_access_key='secret_key',
        endpoint_url='https://s3.amazonaws.com'
    ) as client:
        response = await client.list_buckets()
        print(f"Buckets: {response['Buckets']}")

asyncio.run(test_s3())