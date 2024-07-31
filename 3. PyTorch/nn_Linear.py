import torch

a = torch.rand(256,50)
b = torch.rand(256,50)
c = torch.rand(256,50)

in_features = torch.cat((a,b,c), dim = 1)[0,:]

model = torch.nn.Linear(len(in_features), 768)

out_features = model(in_features).reshape((768,1))
r_vectors = out_features.mm(out_features.t())


h_vectors = torch.rand((1024,768))
t_vectors = torch.rand((1024,768))


logits = (h_vectors.mm(r_vectors)).mm(t_vectors.t())
