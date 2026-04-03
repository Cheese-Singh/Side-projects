import mlx.core as mx

print(
'''\n
    ===1. Working with Columns===
''')
a = mx.array([90, 50, 60, 20, 10])
print(a)
print(a.shape, a.dtype)
a = mx.array([90, 50, 60, 20, 10], mx.int16)
print(a)
a = mx.array([100.1, 50, 60, 20, 10], mx.float64)
print(a)
a = mx.array([90.1, 50, 60, 20, 10])
print(a, "\n")

for x in range(a.size):
    a[x] = a[x] + 11
    print(a[x])

a2 = mx.add(a, 5)
print(a2)
a2 = mx.subtract(a,10.1)
print(a2)
#Same for multiple, divide

a3 = mx.sum(a)
print(a3)
a4 = mx.max(a)
print(a4)
#Same for min, mean

a5 = mx.add(a,5,stream=mx.cpu)
print(a5)
a6 = mx.add(a,1,stream=mx.gpu)
print(a6)

print(
'''\n
    ===2. Working with Rows===
''')
b = mx.array([[10],[2],[7],[14],[6]])
print(b)
print(b.shape, b.dtype)

print(
'''\n
    ===3. Working on Multiple Arrays===
''')
c = mx.array([[8],[16],[11],[4],[12]])
c1 = mx.add(b,c)
print(c1)
#Same for other operators, and for columns

#Vector/Matrix Multiplication
d = mx.array([1,2,3], mx.float32)
e = mx.array([4,5,6], mx.float32)
d1 = mx.matmul(d,e)
print(d1)

d = mx.array([[1,2],[3,4],[5,6]], mx.float32)
e = mx.array([[5,6,7]], mx.float32)
d2 = mx.matmul(e,d)
print(d2)


print(
'''\n
    ===4. Working with Matrices===
''')
a = mx.arange(10,60,10)
b = mx.zeros([3,5])
c = mx.zeros(a.shape)
d = mx.random.randint(0, 100, a.shape)

print(a)
print(b)
print(c)
print(d)

e = mx.array([[[10,20,30],[90,80,70]]])
f = mx.flatten(e)
g = mx.squeeze(e)
h = mx.expand_dims(g,1)
i = mx.concatenate([a,c])
j = mx.reshape(f,(2,3))
k = mx.broadcast_to(f, (6,6))
l = mx.transpose(k)
print(e)
print(f)
print(g)
print(h)
print(i)
print(j)
print(k)
print(l)

print(
'''\n
    ===5. Manual Tokenization===
''')
sentence = "Who is Ada Lovelace"
tokens = sentence.split()
print("Tokens",tokens)

vocab = {
    "Who":1,
    "is":2,
    "Ada":3,
    "Lovelace":4
}

token_indices = [vocab[word] for word in tokens]
print("Token indices",token_indices)

tensor = mx.array(token_indices)
print("Tensor: ",tensor)

print(
'''\n
    ===6. Manually Creating an Embedding Layer===
''')
vocab_size = 128256
hidden_size = 4096

embedding_matrix = mx.random.normal([vocab_size, hidden_size])
print(embedding_matrix)
print(embedding_matrix.shape)

#example input : batch of token indices
input_ids = mx.array([1000,921,100,22])
embeddings = embedding_matrix[input_ids]
print(embeddings.shape)
print(embeddings)