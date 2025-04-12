import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Função de ativação sigmoid: f(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivada da função sigmoid: f'(x) = f(x) * (1 - f(x))"""
    return x * (1 - x)

class SimpleNeuralNetwork:
    def __init__(self):
        # Inicialização dos pesos conforme exemplo
        self.w1, self.w2 = 0.15, 0.20  # Pesos entre entrada e primeira camada oculta
        self.w3, self.w4 = 0.25, 0.30
        self.b1 = 0.35                  # Bias da primeira camada
        
        self.w5, self.w6 = 0.40, 0.45  # Pesos entre camada oculta e saída
        self.w7, self.w8 = 0.50, 0.55
        self.b2 = 0.60                  # Bias da camada de saída
        
        self.learning_rate = 0.5

    def forward_propagation(self, i1, i2):
        """
        Propagação para frente (forward propagation):
        Calcula a saída da rede neural para as entradas i1 e i2
        """
        # Primeira camada
        self.net_h1 = i1 * self.w1 + i2 * self.w2 + self.b1
        self.net_h2 = i1 * self.w3 + i2 * self.w4 + self.b1
        self.g_h1 = sigmoid(self.net_h1)
        self.g_h2 = sigmoid(self.net_h2)

        # Segunda camada
        self.net_o1 = self.g_h1 * self.w5 + self.g_h2 * self.w6 + self.b2
        self.net_o2 = self.g_h1 * self.w7 + self.g_h2 * self.w8 + self.b2
        self.g_o1 = sigmoid(self.net_o1)
        self.g_o2 = sigmoid(self.net_o2)

        return self.g_o1, self.g_o2

    def backward_propagation(self, i1, i2, d1, d2):
        """
        Retropropagação (backpropagation):
        Atualiza os pesos da rede com base no erro
        """
        # Cálculo dos deltas da camada de saída
        delta_o1 = -(d1 - self.g_o1) * sigmoid_derivative(self.g_o1)
        delta_o2 = -(d2 - self.g_o2) * sigmoid_derivative(self.g_o2)

        # Atualização dos pesos da segunda camada
        self.w5 -= self.learning_rate * delta_o1 * self.g_h1
        self.w6 -= self.learning_rate * delta_o1 * self.g_h2
        self.w7 -= self.learning_rate * delta_o2 * self.g_h1
        self.w8 -= self.learning_rate * delta_o2 * self.g_h2

        # Cálculo dos deltas da camada oculta
        delta_h1 = (delta_o1 * self.w5 + delta_o2 * self.w7) * sigmoid_derivative(self.g_h1)
        delta_h2 = (delta_o1 * self.w6 + delta_o2 * self.w8) * sigmoid_derivative(self.g_h2)

        # Atualização dos pesos da primeira camada
        self.w1 -= self.learning_rate * delta_h1 * i1
        self.w2 -= self.learning_rate * delta_h1 * i2
        self.w3 -= self.learning_rate * delta_h2 * i1
        self.w4 -= self.learning_rate * delta_h2 * i2

    def calculate_error(self, d1, d2):
        """Calcula o erro total da rede"""
        error_o1 = 0.5 * (d1 - self.g_o1) ** 2
        error_o2 = 0.5 * (d2 - self.g_o2) ** 2
        return error_o1 + error_o2

    def train(self, i1, i2, d1, d2, epochs):
        """
        Treina a rede neural por um número específico de épocas
        Retorna o histórico de erros para plotagem
        """
        error_history = []
        
        for epoch in range(epochs):
            # Forward propagation
            self.forward_propagation(i1, i2)
            
            # Calcula e armazena o erro
            error = self.calculate_error(d1, d2)
            error_history.append(error)
            
            # Backward propagation
            self.backward_propagation(i1, i2, d1, d2)
            
            # Mostra progresso a cada 5 épocas
            if epoch % 5 == 0:
                print(f"Época {epoch}: Erro = {error:.9f}")
        
        return error_history

# Dados do problema
i1, i2 = 0.05, 0.1      # Entradas
d1, d2 = 0.01, 0.99     # Saídas desejadas
epochs = 30          # Número de épocas

# Criação e treinamento da rede
nn = SimpleNeuralNetwork()
error_history = nn.train(i1, i2, d1, d2, epochs)

# Plotagem do gráfico de erro
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), error_history)
plt.title('Erro vs Épocas de Treinamento')
plt.xlabel('Época')
plt.ylabel('Erro Total')
plt.grid(True)
plt.show()

# Resultados finais
final_outputs = nn.forward_propagation(i1, i2)
final_error = nn.calculate_error(d1, d2)

print("\nResultados Finais:")
print(f"Entradas: i1={i1}, i2={i2}")
print(f"Saídas Desejadas: d1={d1}, d2={d2}")
print(f"Saídas Obtidas: o1={final_outputs[0]:.9f}, o2={final_outputs[1]:.9f}")
print(f"Erro Final: {final_error:.9f}")
