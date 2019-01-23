This projects summarizes the learning outcomes for RNNS. 

The motivation for using Recurrent Neural networks arises in many different applications when dealing with sequential data i.e. there is a temporal component to the data. Such applications include: natural language processing, machine translation and time-series forecasting.
What allows RNNs to capture the essence behing sequential data is the recurrent connections that retains the state of the activation of the hidden units.

Traditional feedforward neural networks are very powerful tools and are being widely used in many applications, however, one of the main limitations is the assumption of independence of the input data, also known as sequence-agnostic, therefore, they are stateless [critical review of RNN]. Contrastyly, if the data is related such as a frame sequence from a video or a sensor data measurements from a dynamical system such as a moving vehicles, then the indpendence assumption is no longer valid.

The basic notation of RNNs: 
The input sequence (x^(1), x^(2), ..., x(T)) and the output is y(T). Where, x and y are vectors. In this specific case we are interested in the current estimate only and not a sequence of estimates in the future.

The training data is setup as an input sequence and output estimate pair. 

Points to cover:

* Primary intuition: 
--
* Shaping the data:
--
* The sequence length parameter:
--
* Creating the model:
using several APIs 
* Application in forecasting:
--



Source:

* https://arxiv.org/pdf/1506.00019.pdf
* https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
* Deep learning course by Andrew Ng
