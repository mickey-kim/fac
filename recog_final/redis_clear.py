import redis

r = redis.StrictRedis()
r2 = redis.StrictRedis(port=6380)
r3 = redis.StrictRedis(port=6381)
r4 = redis.StrictRedis(port=6382)
r5 = redis.StrictRedis(port=6383)

r.flushdb()
r2.flushdb()
r3.flushdb()
r4.flushdb()
r5.flushdb()
