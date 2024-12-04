from pipeline import agent
import matplotlib.pyplot as plt

a = agent()

a.train(lr=0.01, epochs=30)
pred = a.pred()


plt.plot(pred)
plt.show()






