import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from tqdm import tqdm


class KMeansCustom:    
    def __init__(
        self, 
        n_clusters: int = 500,
        max_iter: int = 150,
        tol: float = 1e-6,
        random_state: Optional[int] = 42,
        verbose: bool = True
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        # Variables de estado
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # 1. Elegir primer centroide aleatoriamente
        centroids[0] = X[np.random.randint(n_samples)]
        
        if self.verbose:
            print(f"Inicialización K-Means++: 0/{self.n_clusters}", end="", flush=True)
        
        # 2. Para datasets grandes, usar subset para acelerar
        if n_samples > 50000:
            # Usar subset aleatorio para cálculos de distancia
            subset_size = min(50000, n_samples)
            subset_indices = np.random.choice(n_samples, subset_size, replace=False)
            X_subset = X[subset_indices]
        else:
            X_subset = X
            subset_indices = np.arange(n_samples)
        
        # 3. Elegir siguientes centroides con K-Means++
        for i in range(1, self.n_clusters):
            if self.verbose and i % 50 == 0:
                print(f"\rInicialización K-Means++: {i}/{self.n_clusters}", end="", flush=True)
            
            # Calcular distancias mínimas usando vectorización de NumPy
            distances = np.full(len(X_subset), np.inf)
            for j in range(i):
                dist_to_centroid = np.sum((X_subset - centroids[j])**2, axis=1)
                distances = np.minimum(distances, dist_to_centroid)
            
            # Elegir con probabilidad proporcional a distancia
            probabilities = distances / distances.sum()
            cumulative = np.cumsum(probabilities)
            r = np.random.rand()
            
            # Encontrar índice
            selected_idx = np.searchsorted(cumulative, r)
            if selected_idx >= len(subset_indices):
                selected_idx = len(subset_indices) - 1
            
            # El centroide corresponde al índice original
            original_idx = subset_indices[selected_idx] if n_samples > 50000 else selected_idx
            centroids[i] = X[original_idx]
        
        if self.verbose:
            print(f"\rInicialización K-Means++: {self.n_clusters}/{self.n_clusters} ✓")
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:

        # Usar broadcasting de NumPy para calcular todas las distancias a la vez
        # X: (n_samples, n_features)
        # centroids: (n_clusters, n_features)
        # distances: (n_samples, n_clusters)
        
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # Puntos asignados a cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Nuevo centroide = promedio de puntos asignados
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # Si no hay puntos asignados, mantener centroide anterior
                # (esto puede pasar con K muy grande)
                if hasattr(self, 'cluster_centers_') and self.cluster_centers_ is not None:
                    new_centroids[k] = self.cluster_centers_[k]
                else:
                    # Inicializar aleatoriamente
                    new_centroids[k] = X[np.random.randint(X.shape[0])]
        
        return new_centroids
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        inertia = 0.0
        for i in range(len(X)):
            centroid = centroids[labels[i]]
            inertia += np.linalg.norm(X[i] - centroid) ** 2
        return inertia
    
    def _has_converged(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        if old_centroids is None:
            return False
        
        # Convergencia si el cambio en centroides es menor que tolerancia
        centroid_shift = np.linalg.norm(new_centroids - old_centroids)
        return centroid_shift < self.tol
    
    def fit(self, X: np.ndarray) -> 'KMeansCustom':
        if self.verbose:
            print("Initialization complete")
        
        # 1. Inicializar centroides
        centroids = self._init_centroids(X)
        old_centroids = None
        
        # 2. Algoritmo EM (Expectation-Maximization)
        for iteration in range(self.max_iter):
            # E-step: Asignar puntos a clusters
            labels = self._assign_clusters(X, centroids)
            
            # M-step: Actualizar centroides
            old_centroids = centroids.copy()
            centroids = self._update_centroids(X, labels)
            
            # Calcular inercia
            inertia = self._calculate_inertia(X, labels, centroids)
            
            if self.verbose:
                print(f"Iteration {iteration}, inertia {inertia}.")
            
            # Verificar convergencia
            if self._has_converged(old_centroids, centroids):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations.")
                break
            
            self.n_iter_ = iteration + 1
        
        # Guardar resultados finales
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice clusters para nuevos datos
        """
        if self.cluster_centers_ is None:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero.")
        
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Entrena y predice en un solo paso
        """
        self.fit(X)
        return self.labels_


def compare_with_sklearn():
    """
    Compara nuestra implementación con sklearn para validar correctitud
    """
    # Generar datos de prueba
    np.random.seed(42)
    X = np.random.randn(1000, 13)  # Simular descriptores MFCC
    
    # Nuestra implementación
    print("🔨 Probando K-Means Custom...")
    kmeans_custom = KMeansCustom(n_clusters=10, random_state=42, verbose=False)
    labels_custom = kmeans_custom.fit_predict(X)
    
    # Sklearn para comparación
    print("📚 Probando K-Means sklearn...")
    from sklearn.cluster import KMeans
    kmeans_sklearn = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels_sklearn = kmeans_sklearn.fit_predict(X)
    
    # Comparar resultados
    print(f"\n📊 Resultados:")
    print(f"Custom  - Inercia: {kmeans_custom.inertia_:.2f}, Iteraciones: {kmeans_custom.n_iter_}")
    print(f"Sklearn - Inercia: {kmeans_sklearn.inertia_:.2f}, Iteraciones: {kmeans_sklearn.n_iter_}")
    
    # Los clusters pueden estar en diferente orden, pero la calidad debe ser similar
    inertia_diff = abs(kmeans_custom.inertia_ - kmeans_sklearn.inertia_)
    print(f"Diferencia en inercia: {inertia_diff:.2f}")
    
    if inertia_diff < kmeans_sklearn.inertia_ * 0.1:  # Diferencia < 10%
        print("✅ Implementación correcta: Resultados similares a sklearn")
    else:
        print("⚠️  Posible problema: Gran diferencia con sklearn")


if __name__ == "__main__":
    compare_with_sklearn()
