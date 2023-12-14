import matplotlib.pyplot as plt

def get_graph(history:list,name):
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    
    axes[0].plot(history['loss'], 'b-', label='loss')
    axes[0].plot(history['val_loss'], 'r-', label='val_loss')
    plt.title('monitoring loss')
    plt.xlabel('Epoch')
    axes[1].plot(history['lr'], 'r-', label='lr')
    plt.title('monitoring learning rate')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig('graph/graph_{}.png'.format(name))
    return

def visualization_metrix(model,x,y,name):
    pred = model(x)
    plt.scatter(pred, y, color = 'red')
    plt.plot(pred, y, color = 'blue')
    plt.show()
    plt.savefig('graph/visualization_{}.png'.format(name))
    return