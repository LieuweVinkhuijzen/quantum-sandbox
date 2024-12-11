import minorminer as minor

triangle = [(0,1), (1, 2), (2,0)]
square = [(0, 1), (1, 2), (2,3), (3,0)]

embedding = minor.find_embedding(triangle, square)
print(len(embedding))
print(f'type: {type(embedding)}')
print(embedding)

# yields a dictionary
# dict [int -> list[int]]
# mapping each vertex of 'triangle' to one or more vertices in 'square', s.t.
# for each edge (u,v) in triangle, there is x in embedding[u] and y in embedding[v] such that (x,y) in square