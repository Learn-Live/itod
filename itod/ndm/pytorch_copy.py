# https://discuss.pytorch.org/t/deep-copying-pytorch-modules/13514
mymodel = ()
model_copy = type(mymodel)()  # get a new instance
model_copy.load_state_dict(mymodel.state_dict())  # copy weights and stuff
