def weighted_metric(metric: callable, y_true, y_pred):
  assert y_true.shape == y_pred.shape
  assert y_true.shape[-1] == 4
  return (metric(y_true[:,0],y_pred[:,0]) +
          metric(y_true[:,1],y_pred[:,1]) + 
          metric(y_true[:,2],y_pred[:,2]) + 
          metric(y_true[:,3],y_pred[:,3]))/4