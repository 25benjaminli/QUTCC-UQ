import torch.nn as nn

from models.im2im.quantile_layer import QuantileRegressionLayer

class ModelWithUncertainty(nn.Module):
  def __init__(self, baseModel, last_layer, params):
      super(ModelWithUncertainty, self).__init__()
      self.baseModel = baseModel
      self.last_layer = last_layer
      self.params = params
  
  def forward(self, x, timesteps=None):
      if timesteps is not None:
          x = self.baseModel(x, timesteps=timesteps)
      else:
          x = self.baseModel(x)
      return self.last_layer(x)

def add_uncertainty(model, params, out_channels=1): 
  last_layer = None
  
  if params["uncertainty_type"] == "quantiles":
    #This is the one we are concerned with
    # print('Adding quantile regression layer')
    last_layer = QuantileRegressionLayer(32, out_channels, params) 
   
  return ModelWithUncertainty(model, last_layer, params)
