import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

# Crear el entorno Frozen Lake
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

# Parámetros del Q-learning optimizados
alpha = 0.1  # Tasa de aprendizaje reducida para mayor estabilidad
gamma = 0.99  # Factor de descuento aumentado para valorar más las recompensas futuras
epsilon = 1.0  # Tasa de exploración inicial
epsilon_min = 0.01
epsilon_decay = 0.995

# Inicializar la tabla Q con valores pequeños aleatorios para romper empates
Q = np.random.uniform(low=0, high=0.1, size=(env.observation_space.n, env.action_space.n))

# Parámetros de entrenamiento ajustados
num_episodes = 5000  # Reducido a 5000 episodios
max_steps = 100  # Mantenemos 100 pasos máximos por episodio

# Listas para seguimiento
rewards = []
steps_per_episode = []
success_count = 0
last_100_rewards = []

# Entrenamiento del agente
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False
    
    for step in range(max_steps):
        # Selección de acción epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        
        # Ejecutar acción
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Actualización Q-learning con clip para estabilidad
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )
        
        total_reward += reward
        state = next_state
        
        if done or truncated:
            if reward == 1:
                success_count += 1
            break
    
    # Actualizar métricas
    rewards.append(total_reward)
    steps_per_episode.append(step)
    last_100_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
    
    # Actualizar epsilon con decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Mostrar progreso cada 100 episodios
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(last_100_rewards)
        success_rate = success_count / min(episode + 1, 100) * 100
        print(f"Episodio: {episode + 1}")
        print(f"Tasa de éxito (últimos 100): {success_rate:.2f}%")
        print(f"Recompensa promedio: {avg_reward:.3f}")
        print(f"Epsilon: {epsilon:.3f}")
        success_count = 0

# Función de prueba mejorada
def test_agent(env, Q, num_episodes=5):
    success_count = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        print(f"\nEpisodio de prueba {episode + 1}")
        time.sleep(1)  # Pausa entre episodios
        
        for step in range(100):  # Límite de 100 pasos
            action = np.argmax(Q[state, :])
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            total_steps += 1
            
            time.sleep(0.5)  # Pausa para visualización
            
            if done:
                if reward == 1:
                    success_count += 1
                    print("¡Éxito! Meta alcanzada")
                else:
                    print("Falló - Caída en agujero")
                break
    
    success_rate = (success_count / num_episodes) * 100
    avg_steps = total_steps / num_episodes
    
    print(f"\nResultados finales:")
    print(f"Tasa de éxito: {success_rate:.2f}%")
    print(f"Pasos promedio: {avg_steps:.2f}")

# Graficar resultados
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
plt.title('Recompensa Promedio (Media Móvil 100 episodios)')
plt.xlabel('Episodio')
plt.ylabel('Recompensa')

plt.subplot(1, 2, 2)
plt.plot(np.convolve(steps_per_episode, np.ones(100)/100, mode='valid'))
plt.title('Pasos Promedio (Media Móvil 100 episodios)')
plt.xlabel('Episodio')
plt.ylabel('Pasos')
plt.tight_layout()
plt.show()

# Probar el agente entrenado
print("\nProbando el agente entrenado:")
test_agent(env, Q)

# Guardar la política aprendida
np.save('qtable_frozen_lake.npy', Q)
print("\nTabla Q guardada como 'qtable_frozen_lake.npy'")

env.close()
