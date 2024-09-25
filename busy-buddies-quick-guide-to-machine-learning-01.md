### Machine Learning and Deep Learning Lecture Notes

#### **Chapter 1: Linear Regression & Logistic Regression**

- **1.1 Linear Regression**
    - **Goal**: Predict continuous values using a linear equation.
    - **Key Formula**:
      $$ y = \beta_0 + \beta_1x_1 + \dots + \beta_nx_n + \epsilon $$
      - \(y\): predicted value  
      - \(\beta\): coefficients (weights)  
      - \(x\): input features  
      - \(\epsilon\): error term
    - **Loss Function**:  
      Mean Squared Error (MSE):
      $$ L = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y_i})^2 $$
    - **Gradient Descent Formula**:
      $$ \beta_j = \beta_j - \eta \cdot \frac{\partial L}{\partial \beta_j} $$
    - **Partial Derivative Formula**:
      $$ \frac{\partial L}{\partial \beta_j} = -\frac{2}{N} \sum_{i=1}^{N} x_{ij}(y_i - \hat{y_i}) $$
      
- **1.2 Logistic Regression**
    - **Goal**: Solve classification problems with discrete outputs (0 or 1).
    - **Key Formula**:
      $$ p(y=1|x) = \frac{1}{1+e^{-z}}, \quad z = w^Tx + b $$
    - **Loss Function**:  
      Cross-entropy loss:
      $$ L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y_i}) + (1 - y_i)\log(1 - \hat{y_i}) \right] $$

---

#### **Chapter 2: Fully Connected Networks & Backpropagation**

- **2.1 Fully Connected Networks (FC)**
    - Every neuron is connected to all neurons in the previous layer.
    - **Calculation Formula**:
      $$ z = W^T x + b $$
      - \(W\): weight matrix  
      - \(x\): input vector  
      - \(b\): bias term
    - **Activation Functions**:
      - ReLU:
        $$ f(x) = \max(0, x) $$
      - Sigmoid:
        $$ f(x) = \frac{1}{1+e^{-x}} $$
      
- **2.2 Backpropagation**
    - **Goal**: Update network weights by propagating errors backward.
    - **Loss Function**:  
      Examples: MSE or cross-entropy.
    - **Chain Rule**:
      $$ \frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial W_i} $$
    - **Gradient Update Formula**:
      $$ W = W - \eta \frac{\partial L}{\partial W} $$

---

#### **Chapter 3: Convolutional Neural Networks (CNN)**

- **3.1 Convolution Operation**
    - Convolution layers apply filters (weight matrices) to input data to extract local features.
    - **Key Formula**:
      $$ (f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau $$
    - **Feature Map**:  
      The result of applying convolution.

- **3.2 Pooling Operation**
    - **Max Pooling**: Selects the maximum value from a local region, reducing the data's dimensionality.
    - **Average Pooling**: Selects the average value from a local region.
    
- **3.3 CNN Architecture**
    - Common structure:  
      Convolution layer -> Activation layer -> Pooling layer -> Fully connected layer
    - **Use Cases**: Image classification, object detection, etc.

---

#### **Chapter 4: Recurrent Neural Networks, LSTM & GRU**

- **4.1 Recurrent Neural Networks (RNN)**
    - **Goal**: Process sequential data by maintaining information from previous time steps.
    - **Key Formula**:
      $$ h_t = f(W_{hh} h_{t-1} + W_{xx} x_t) $$
      - \(h_t\): hidden state at time \(t\)  
      - \(x_t\): input at time \(t\)  
      - \(W_{hh}\): hidden state weight matrix  
      - \(W_{xx}\): input weight matrix
    - **Problem**: Gradient vanishing over long sequences.

- **4.2 Long Short-Term Memory (LSTM)**
    - **Goal**: Solve the gradient vanishing problem in RNNs by introducing gate mechanisms.
    - **Three Gates**:  
      - **Input Gate**: Controls the amount of new information added.
      - **Forget Gate**: Controls the amount of old information to forget.
      - **Output Gate**: Controls the amount of information to output.
    - **Key Formula**:
      - Cell state update:
        $$ c_t = f_t * c_{t-1} + i_t * \tilde{c}_t $$
      - Hidden state:
        $$ h_t = o_t * \tanh(c_t) $$

- **4.3 Gated Recurrent Unit (GRU)**
    - **Goal**: Simplify LSTM by using two gates (reset and update gates).
    - **Reset Gate**: Determines whether to ignore information from the previous time step.
    - **Update Gate**: Controls the degree of state update at the current time step.
