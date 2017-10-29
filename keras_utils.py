import matplotlib.pyplot as plt

def draw_error_validation_surface(history):
    ''' usage: 
    fit_stats = model.fit()
    draw_error_surface(fit_stats.history)
    '''
    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()

def draw_error_surface(history):
    ''' usage: 
    fit_stats = model.fit()
    draw_error_surface(fit_stats.history)
    '''
    # summarize history for loss
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
