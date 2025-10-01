import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation
import sympy as sp
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

# CONFIGURAÇÕES E CONSTANTES

@dataclass
class PlotConfig:
    """Configurações de plotagem"""
    # Intervalos
    INTERVAL_1D: Tuple[float, float] = (-5, 5)
    INTERVAL_2D: Tuple[float, float] = (-3, 3)
    
    # Resoluções
    RESOLUTION_1D: int = 400
    RESOLUTION_2D: int = 100
    RESOLUTION_3D: int = 80
    
    # Cores
    COLOR_FUNCTION: str = 'C0'
    COLOR_POINT: str = 'red'
    COLOR_TANGENT: str = 'cyan'
    COLOR_LIMIT_RIGHT: str = 'blue'
    COLOR_LIMIT_LEFT: str = 'orange'
    COLOR_PANEL: str = 'lightblue'
    
    # Tamanhos
    MARKER_SIZE_MAIN: int = 10
    MARKER_SIZE_LATERAL: int = 8/2
    LINE_WIDTH: int = 2
    
    # Posições UI (x, y, largura, altura)
    POS_TEXTBOX_FUNC: Tuple[float, float, float, float] = (0.1, 0.15, 0.50, 0.04)
    POS_TEXTBOX_POINT: Tuple[float, float, float, float] = (0.1, 0.09, 0.50, 0.04)
    POS_SLIDER_Z: Tuple[float, float, float, float] = (0.65, 0.09, 0.13, 0.03)
    POS_BUTTON_ANIMATE: Tuple[float, float, float, float] = (0.65, 0.15, 0.06, 0.04)
    POS_CHECK_ANIMATE: Tuple[float, float, float, float] = (0.72, 0.13, 0.06, 0.09)
    
    # Painel de informações
    PANEL_X: float = 0.81
    PANEL_Y_START: float = 0.7
    PANEL_Y_STEP: float = 0.06
    
    # Animação
    ANIMATION_STEPS: int = 200
    ANIMATION_INTERVAL: int = 50  # ms
    EPSILON_START: float = .1
    EPSILON_END: float = 1e-6
    # Mantém marcadores finais por X ms após fim da animação
    HOLD_AFTER_ANIMATION_MS: int = 1500
    # Zoom durante animação: janela inicial e mínima (meia-janela)
    ZOOM_START_WINDOW: float = 1.0
    ZOOM_MIN_WINDOW: float = 1e-2
    
    # Detecção de descontinuidades
    DISCONTINUITY_THRESHOLD: float = 10.0
    
    # Percentis para ajuste de eixo Y
    PERCENTILE_LOW: float = 5
    PERCENTILE_HIGH: float = 95
    MARGIN_Y: float = 0.5

config = PlotConfig()

# Símbolos matemáticos
x, y, z = sp.symbols("x y z")

class MathEngine:
    """Motor de cálculos matemáticos"""
    
    @staticmethod
    def parse_function(func_str: str) -> sp.Expr:
        """Converte string para expressão SymPy"""
        try:
            return sp.sympify(func_str)
        except:
            raise ValueError(f"Não foi possível interpretar: {func_str}")
    
    @staticmethod
    def make_numeric(function: sp.Expr, vars_: List[sp.Symbol]) -> callable:
        """Converte expressão simbólica para função numérica"""
        return sp.lambdify(vars_, function, "numpy")
    
    @staticmethod
    def calculate_limit(f: sp.Expr, var: sp.Symbol, point: float, 
                       direction: str = 'both') -> Tuple[Any, bool]:
        """
        Calcula limite em um ponto
        
        Args:
            f: Função simbólica
            var: Variável
            point: Ponto de avaliação
            direction: 'right', 'left', ou 'both'
        
        Returns:
            (valor_limite, existe_bool)
        """
        try:
            if direction == 'right':
                lim = sp.limit(f, var, point, '+')
            elif direction == 'left':
                lim = sp.limit(f, var, point, '-')
            else:
                lim_maior = sp.limit(f, var, point, '+')
                lim_menor = sp.limit(f, var, point, '-')

                if lim_maior != lim_menor:
                    return "Não existe", False

                lim = lim_maior
            
            if lim == sp.oo:
                return "+∞", False
            elif lim == -sp.oo:
                return "-∞", False
            elif lim.is_finite:
                return float(lim), True
            else:
                return "Não existe", False
        except:
            return "Indefinido", False
    
    @staticmethod
    def calculate_derivative(f: sp.Expr, vars_: List[sp.Symbol], 
                           point: Tuple[float, ...]) -> Tuple[Any, Any]:
        """
        Calcula derivada(s) no ponto
        
        Returns:
            Para 1D: (df_expr, df_value)
            Para 2D/3D: ((df_dx, df_dy), (val_x, val_y))
        """
        try:
            if len(vars_) == 1:
                df = sp.diff(f, vars_[0])
                df_val = df.subs(vars_[0], point[0])
                return df, float(df_val) if df_val.is_finite else None
            else:
                df_dx = sp.diff(f, vars_[0])
                df_dy = sp.diff(f, vars_[1])
                
                subs_dict = [(vars_[0], point[0]), (vars_[1], point[1])]
                if len(point) > 2 and len(vars_) > 2:
                    subs_dict.append((vars_[2], point[2]))
                
                df_dx_val = df_dx.subs(subs_dict)
                df_dy_val = df_dy.subs(subs_dict)
                
                return (df_dx, df_dy), (
                    float(df_dx_val) if df_dx_val.is_finite else None,
                    float(df_dy_val) if df_dy_val.is_finite else None
                )
        except:
            return None, None
    
    @staticmethod
    def evaluate_at_point(f: sp.Expr, vars_: List[sp.Symbol], 
                         point: Tuple[float, ...]) -> Optional[float]:
        """Avalia função em um ponto"""
        try:
            subs_dict = list(zip(vars_, point))
            result = f.subs(subs_dict)
            return float(result) if result.is_finite else None
        except:
            return None

# DETECÇÃO DE DESCONTINUIDADES

class DiscontinuityDetector:
    """Detecta descontinuidades em funções 1D"""
    
    @staticmethod
    def detect_segments(xs: np.ndarray, ys: np.ndarray, 
                       threshold_multiplier: float = config.DISCONTINUITY_THRESHOLD
                       ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Divide função em segmentos contínuos
        
        Returns:
            Lista de tuplas (x_segment, y_segment)
        """
        segments = []
        current_x = []
        current_y = []
        
        # Calcula threshold baseado na variação total
        valid_ys = ys[np.isfinite(ys)]
        if len(valid_ys) < 2:
            return [(xs, ys)]
        
        y_range = np.ptp(valid_ys)  # peak-to-peak
        threshold = threshold_multiplier * y_range / len(xs)
        
        for i in range(len(xs)):
            if np.isfinite(ys[i]):
                if current_y and abs(ys[i] - current_y[-1]) > threshold:
                    # Salto grande detectado - novo segmento
                    if len(current_x) > 1:
                        segments.append((np.array(current_x), np.array(current_y)))
                    current_x = [xs[i]]
                    current_y = [ys[i]]
                else:
                    current_x.append(xs[i])
                    current_y.append(ys[i])
            else:
                # Valor infinito/NaN - finaliza segmento
                if len(current_x) > 1:
                    segments.append((np.array(current_x), np.array(current_y)))
                current_x = []
                current_y = []
        
        # Adiciona último segmento
        if len(current_x) > 1:
            segments.append((np.array(current_x), np.array(current_y)))
        
        return segments if segments else [(xs[np.isfinite(ys)], ys[np.isfinite(ys)])]

# ============================================================================
# VISUALIZAÇÃO (PLOTS)
# ============================================================================

class PlotRenderer:
    """Renderiza gráficos e visualizações"""
    
    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self.artists = []  # Lista de objetos gráficos para limpeza
    
    def clear_artists(self):
        """Remove todos os artistas (marcadores, textos, etc)"""
        for artist in self.artists:
            try:
                artist.remove()
            except (NotImplementedError, ValueError, AttributeError):
                pass
        self.artists.clear()
    
    def plot_1d_function(self, f_num: callable, f_expr: sp.Expr):
        """Plota função 1D"""
        xs = np.linspace(*config.INTERVAL_1D, config.RESOLUTION_1D)
        ys = f_num(xs)
        
        # Detecta e plota segmentos
        segments = DiscontinuityDetector.detect_segments(xs, ys)
        for seg_x, seg_y in segments:
            self.ax.plot(seg_x, seg_y, linewidth=config.LINE_WIDTH, 
                        color=config.COLOR_FUNCTION)
        
        # Configura eixos
        self.ax.set_title(f"$f(x) = {sp.latex(f_expr)}$", fontsize=12)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True, alpha=0.3)
        
        # Ajusta limites Y
        valid_ys = ys[np.isfinite(ys)]
        if len(valid_ys) > 0:
            y_min, y_max = np.percentile(valid_ys, 
                                        [config.PERCENTILE_LOW, config.PERCENTILE_HIGH])
            y_range = y_max - y_min
            if y_range > 0:
                margin = config.MARGIN_Y * y_range
                self.ax.set_ylim(y_min - margin, y_max + margin)
    
    def plot_constant_function(self, const_val: float, f_expr: sp.Expr):
        """Plota função constante como linha horizontal"""
        xs = np.linspace(*config.INTERVAL_1D, config.RESOLUTION_1D)
        ys = np.full_like(xs, const_val)
        
        self.ax.plot(xs, ys, linewidth=config.LINE_WIDTH, 
                    color=config.COLOR_FUNCTION)
        self.ax.set_title(f"$f(x) = {sp.latex(f_expr)} = {const_val:.6g}$", fontsize=12)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True, alpha=0.3)
        
        # Ajusta limites Y
        y_margin = max(abs(const_val) * 0.2, 1)
        self.ax.set_ylim(const_val - y_margin, const_val + y_margin)
    
    def plot_2d_surface(self, f_num: callable, f_expr: sp.Expr):
        """Plota superfície 2D"""
        xs = np.linspace(*config.INTERVAL_2D, config.RESOLUTION_2D)
        ys = np.linspace(*config.INTERVAL_2D, config.RESOLUTION_2D)
        X, Y = np.meshgrid(xs, ys)
        
        try:
            Z = f_num(X, Y)
            Z = np.nan_to_num(Z, nan=np.nan, posinf=np.nan, neginf=np.nan)
            
            self.ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
            self.ax.set_title(f"$f(x,y) = {sp.latex(f_expr)}$", fontsize=12)
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("f(x,y)")
        except Exception as e:
            self.ax.text2D(0.5, 0.5, f"Erro: {str(e)[:50]}",
                          ha="center", va="center", transform=self.ax.transAxes)
    
    def plot_3d_slice(self, f_num: callable, f_expr: sp.Expr, z_val: float):
        """Plota fatia de função 3D"""
        xs = np.linspace(*config.INTERVAL_2D, config.RESOLUTION_3D)
        ys = np.linspace(*config.INTERVAL_2D, config.RESOLUTION_3D)
        X, Y = np.meshgrid(xs, ys)
        
        try:
            Z = f_num(X, Y, z_val)
            Z = np.nan_to_num(Z, nan=np.nan, posinf=np.nan, neginf=np.nan)
            
            self.ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.9)
            self.ax.set_title(f"$f(x,y,z={z_val:.2f}) = {sp.latex(f_expr)}$", fontsize=12)
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel(f"f(x,y,{z_val:.2f})")
        except Exception as e:
            self.ax.text2D(0.5, 0.5, f"Erro: {str(e)[:50]}",
                          ha="center", va="center", transform=self.ax.transAxes)
    
    def draw_point_marker(self, x_val: float, y_val: float, 
                         filled: bool = True, color: str = None, 
                         size: int = None):
        """Desenha marcador de ponto"""
        color = color or config.COLOR_POINT
        size = size or config.MARKER_SIZE_MAIN
        face = color if filled else 'none'
        
        marker, = self.ax.plot(x_val, y_val, 'o', markersize=size,
                              markerfacecolor=face, markeredgecolor=color,
                              markeredgewidth=config.LINE_WIDTH, zorder=5)
        self.artists.append(marker)
    
    def draw_tangent_line(self, x_val: float, f_val: float, df_val: float):
        """Desenha reta tangente"""
        x_range = self.ax.get_xlim()
        x_tang = np.linspace(x_range[0], x_range[1], 100)
        y_tang = f_val + df_val * (x_tang - x_val)
        
        line, = self.ax.plot(x_tang, y_tang, '--', 
                            color=config.COLOR_TANGENT,
                            alpha=0.7, linewidth=config.LINE_WIDTH,
                            label='Tangente', zorder=4)
        self.artists.append(line)
    
    def draw_tangent_plane(self, point: Tuple[float, float], f_val: float,
                          grad: Tuple[float, float], delta: float = 0.5):
        """Desenha plano tangente 3D"""
        x_val, y_val = point
        fx, fy = grad
        
        x_tang = np.array([x_val - delta, x_val + delta])
        y_tang = np.array([y_val - delta, y_val + delta])
        X_tang, Y_tang = np.meshgrid(x_tang, y_tang)
        Z_tang = f_val + fx*(X_tang - x_val) + fy*(Y_tang - y_val)
        
        surf = self.ax.plot_surface(X_tang, Y_tang, Z_tang,
                                    color=config.COLOR_TANGENT,
                                    alpha=0.5, linewidth=0)
        self.artists.append(surf)
    
    def draw_3d_point(self, x: float, y: float, z: float):
        """Desenha ponto em gráfico 3D"""
        marker = self.ax.scatter([x], [y], [z], c=config.COLOR_POINT,
                                s=100, marker='o', edgecolors='darkred',
                                linewidths=config.LINE_WIDTH, zorder=5)
        self.artists.append(marker)

# ============================================================================
# PAINEL DE INFORMAÇÕES
# ============================================================================

class InfoPanel:
    """Painel lateral com informações matemáticas"""
    
    def __init__(self, fig):
        self.fig = fig
        self.texts = []
    
    def clear(self):
        """Limpa todos os textos"""
        for text in self.texts:
            try:
                text.remove()
            except:
                pass
        self.texts.clear()
    
    def display_1d_info(self, point: float, limits: Dict, derivative: Tuple):
        """Exibe informações para função 1D"""
        lines = [
            f"Ponto: $x = {point:.3f}$",
            "",
            f"Lim $x \\to {point:.3f}^+$: {limits['right']}",
            f"Lim $x \\to {point:.3f}^-$: {limits['left']}",
            f"Limite Global: {limits['global']}",
            ""
        ]
        
        df_expr, df_val = derivative
        if df_val is not None:
            lines.append(f"$f'(x) = {sp.latex(df_expr)}$")
            lines.append(f"$f'({point:.3f}) = {df_val:.3f}$")
        else:
            lines.append("Derivada: Não existe")
        
        self._render_lines(lines)
    
    def display_constant_info(self, point: float, const_val: float, f_expr: sp.Expr):
        """Exibe informações para função constante"""
        lines = [
            f"Ponto: $x = {point:.3f}$",
            "",
            f"Função constante: $f(x) = {sp.latex(f_expr)}$",
            f"Valor: ${const_val:.6g}$",
            "",
            "Derivada: $f'(x) = 0$"
        ]
        self._render_lines(lines)
    
    def display_2d_info(self, point: Tuple[float, float], derivatives: Tuple):
        """Exibe informações para função 2D"""
        x_val, y_val = point
        lines = [f"Ponto: $({x_val:.2f}, {y_val:.2f})$", ""]
        
        derivs, deriv_vals = derivatives
        if deriv_vals and deriv_vals[0] is not None:
            lines.extend([
                f"$\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(derivs[0])}$",
                f"$\\left.\\frac{{\\partial f}}{{\\partial x}}\\right|_p = {deriv_vals[0]:.3f}$",
                "",
                f"$\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(derivs[1])}$",
                f"$\\left.\\frac{{\\partial f}}{{\\partial y}}\\right|_p = {deriv_vals[1]:.3f}$"
            ])
        else:
            lines.append("Derivadas: Não existem")
        
        self._render_lines(lines)
    
    def _render_lines(self, lines: List[str]):
        """Renderiza linhas de texto no painel"""
        y_pos = config.PANEL_Y_START
        for i, line in enumerate(lines):
            bbox = dict(boxstyle='round', facecolor=config.COLOR_PANEL, 
                       alpha=0.8) if i == 0 else None
            
            text = self.fig.text(config.PANEL_X, y_pos, line,
                               fontsize=9, verticalalignment='top',
                               bbox=bbox)
            self.texts.append(text)
            y_pos -= config.PANEL_Y_STEP

# ============================================================================
# ANIMAÇÃO DE LIMITES
# ============================================================================



class LimitAnimator:
    """Anima aproximação de limites laterais com diversos modos.

    Modos suportados:
      - "Zoom": aproxima visualmente os eixos para o ponto.
      - "Trace": desenha o traço dos pontos aproximados.
      - "Pulse": faz os marcadores pulsarem durante a animação.
      - "Hold": mantém os marcadores por um tempo após terminar.
    """

    def __init__(self, ax: plt.Axes, fig: plt.Figure, modes: Optional[set] = None) -> None:
        self.ax: plt.Axes = ax
        self.fig: plt.Figure = fig
        self.animation: Optional[FuncAnimation] = None
        self.is_running: bool = False
        self.artists: List[Any] = []
        self._stop_timer: Optional[Any] = None
        self._hold_timer: Optional[Any] = None
        self.modes: set = set(modes) if modes is not None else set()
        self._trace_left: List[Tuple[float, float]] = []
        self._trace_right: List[Tuple[float, float]] = []

    def animate_limit_approach(self,
                               f: sp.Expr,
                               var: sp.Symbol,
                               point: float,
                               direction: str = 'both') -> None:
        if self.is_running:
            return

        self.is_running = True
        f_num = sp.lambdify(var, f, "numpy")
        frames: int = config.ANIMATION_STEPS
        # use logspace to approach epsilon more naturally
        epsilons = np.logspace(np.log10(config.EPSILON_START), np.log10(config.EPSILON_END), frames)

        # resetar os tracejados
        self._trace_left.clear()
        self._trace_right.clear()

        def update(frame: int) -> List[Any]:
            # remover as coisa
            for artist in list(self.artists):
                try:
                    artist.remove()
                except Exception:
                    pass
            self.artists.clear()

            eps = float(epsilons[frame])
            t = frame / max(1, frames - 1)

            # comptamos o zoom pra acabar na metade
            half_window = None
            if 'Zoom' in self.modes:
                half_window = max(config.ZOOM_MIN_WINDOW,
                                  config.ZOOM_START_WINDOW * (1 - t*2))

            # limite do maior
            if direction in ('right', 'both'):
                x_right = float(point + eps)
                try:
                    y_right = float(f_num(x_right))
                    if np.isfinite(y_right):
                        size = config.MARKER_SIZE_LATERAL
                        if 'Pulse' in self.modes:
                            size = config.MARKER_SIZE_LATERAL * (1.0 + 0.5 * np.sin(2 * np.pi * 3 * t))
                        pt, = self.ax.plot(x_right, y_right, 'o',
                                           color=config.COLOR_LIMIT_RIGHT,
                                           markersize=size, alpha=0.95, zorder=6)
                        ln, = self.ax.plot([x_right, x_right],
                                           [self.ax.get_ylim()[0], y_right],
                                           '--', color=config.COLOR_LIMIT_RIGHT,
                                           alpha=0.4, linewidth=1)
                        txt = self.ax.text(x_right, y_right + 0.05 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]),
                                           f'$\\varepsilon={eps:.3e}\\quad f={y_right:.3f}$',
                                           fontsize=8, ha='center',
                                           color=config.COLOR_LIMIT_RIGHT)
                        self.artists.extend([pt, ln, txt])
                        if 'Trace' in self.modes:
                            self._trace_right.append((x_right, y_right))
                            txs, tys = zip(*self._trace_right)
                            trace_line, = self.ax.plot(txs, tys, '-', linewidth=1, alpha=0.6,
                                                       color=config.COLOR_LIMIT_RIGHT, zorder=5)
                            self.artists.append(trace_line)
                except Exception:
                    pass

            # limite do menor
            if direction in ('left', 'both'):
                x_left = float(point - eps)
                try:
                    y_left = float(f_num(x_left))
                    if np.isfinite(y_left):
                        size = config.MARKER_SIZE_LATERAL
                        if 'Pulse' in self.modes:
                            size = config.MARKER_SIZE_LATERAL * (1.0 + 0.5 * np.cos(2 * np.pi * 3 * t))
                        pt, = self.ax.plot(x_left, y_left, 'o',
                                           color=config.COLOR_LIMIT_LEFT,
                                           markersize=size, alpha=0.95, zorder=6)
                        ln, = self.ax.plot([x_left, x_left],
                                           [self.ax.get_ylim()[0], y_left],
                                           '--', color=config.COLOR_LIMIT_LEFT,
                                           alpha=0.4, linewidth=1)
                        txt = self.ax.text(x_left, y_left + 0.05 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]),
                                           f'$\\varepsilon={eps:.3e}\\quad f={y_left:.3f}$',
                                           fontsize=8, ha='center',
                                           color=config.COLOR_LIMIT_LEFT)
                        self.artists.extend([pt, ln, txt])
                        if 'Trace' in self.modes:
                            self._trace_left.append((x_left, y_left))
                            txs, tys = zip(*self._trace_left)
                            trace_line, = self.ax.plot(txs, tys, '-', linewidth=1, alpha=0.6,
                                                       color=config.COLOR_LIMIT_LEFT, zorder=5)
                            self.artists.append(trace_line)
                except Exception:
                    pass

            # apply zoom if requested
            if half_window is not None:
                try:
                    ys_visible = []
                    if self._trace_left:
                        ys_visible.extend([p[1] for p in self._trace_left])
                    if self._trace_right:
                        ys_visible.extend([p[1] for p in self._trace_right])
                    if ys_visible:
                        y_center = float(np.mean(ys_visible))
                        y_half = max(0.2, max(abs(y - y_center) for y in ys_visible))
                        self.ax.set_ylim(y_center - y_half - 0.2, y_center + y_half + 0.2)
                    self.ax.set_xlim(point - half_window, point + half_window)
                except Exception:
                    pass

            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass

            return self.artists

        # create animation
        try:
            self.animation = FuncAnimation(self.fig, update, frames=frames,
                                           interval=config.ANIMATION_INTERVAL, blit=False, repeat=False)
        except Exception as e:
            print(f"[LimitAnimator] Falha ao criar FuncAnimation: {e}")
            self.is_running = False
            return

        if getattr(self.animation, 'event_source', None) is None:
            try:
                timer = self.fig.canvas.new_timer(interval=config.ANIMATION_INTERVAL)
                timer.add_callback(self.animation._step)
                self.animation.event_source = timer
            except Exception as e:
                print(f"[LimitAnimator] Não foi possível criar event_source: {e}")
                self.animation = None
                self.is_running = False
                return

        try:
            self.animation.event_source.start()
        except Exception as e:
            print(f"[LimitAnimator] Falha ao iniciar event_source: {e}")
            self.is_running = False
            return

        total_ms = int(frames * config.ANIMATION_INTERVAL)
        try:
            if getattr(self, '_stop_timer', None) is not None:
                try:
                    self._stop_timer.stop()
                except Exception:
                    pass
            self._stop_timer = self.fig.canvas.new_timer(interval=total_ms + 50)
            self._stop_timer.add_callback(self._on_animation_end)
            self._stop_timer.start()
        except Exception as e:
            print(f"[LimitAnimator] Não foi possível agendar stop_timer: {e}")

    def _on_animation_end(self) -> None:
        try:
            if self.animation and getattr(self.animation, 'event_source', None) is not None:
                try:
                    self.animation.event_source.stop()
                except Exception:
                    pass
        except Exception:
            pass

        if 'Hold' in self.modes:
            try:
                if getattr(self, '_hold_timer', None) is not None:
                    try:
                        self._hold_timer.stop()
                    except Exception:
                        pass
                self._hold_timer = self.fig.canvas.new_timer(interval=config.HOLD_AFTER_ANIMATION_MS)
                self._hold_timer.add_callback(self._final_cleanup)
                self._hold_timer.start()
                self.is_running = False
                return
            except Exception as e:
                print(f"[LimitAnimator] Falha ao agendar hold timer: {e}")

        self._final_cleanup()

    def _final_cleanup(self) -> None:
        for artist in list(self.artists):
            try:
                artist.remove()
            except Exception:
                pass
        self.artists.clear()

        if getattr(self, '_stop_timer', None) is not None:
            try:
                self._stop_timer.stop()
            except Exception:
                pass
            self._stop_timer = None

        if getattr(self, '_hold_timer', None) is not None:
            try:
                self._hold_timer.stop()
            except Exception:
                pass
            self._hold_timer = None

        self.is_running = False
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def stop(self) -> None:
        if self.animation and getattr(self.animation, 'event_source', None) is not None:
            try:
                self.animation.event_source.stop()
            except Exception:
                pass

        if getattr(self, '_stop_timer', None) is not None:
            try:
                self._stop_timer.stop()
            except Exception:
                pass
            self._stop_timer = None

        if getattr(self, '_hold_timer', None) is not None:
            try:
                self._hold_timer.stop()
            except Exception:
                pass
            self._hold_timer = None

        for artist in list(self.artists):
            try:
                artist.remove()
            except Exception:
                pass
        self.artists.clear()

        self.is_running = False
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

class FunctionPlotter:
    """Aplicação principal de plotagem"""
    
    def __init__(self):
        # Estado
        self.current_func: Optional[sp.Expr] = None
        self.current_vars: List[sp.Symbol] = []
        self.selected_point: Optional[Tuple] = None
        self.current_dimension: int = 1
        self.z_value: float = 1.0
        self.animate_limits: bool = False
        
        # Componentes
        self.math_engine = MathEngine()
        self.fig = plt.figure(figsize=(13, 7))
        self.ax = self.fig.add_subplot(111)
        self.renderer = PlotRenderer(self.ax, self.fig)
        self.info_panel = InfoPanel(self.fig)
        # modos de animação selecionáveis
        self.animation_modes: set = set()
        self.animator = LimitAnimator(self.ax, self.fig, modes=self.animation_modes)

        # Widgets
        self._create_widgets()
        
        # Conecta eventos
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Função padrão
        self._plot_function("(x**2 - 1)/(x-1)")
    
    def _create_widgets(self):
        """Cria interface de usuário"""
        # Ajusta layout
        plt.subplots_adjust(left=0.1, right=0.78, bottom=0.25)
        
        # TextBox função
        ax_func = plt.axes(config.POS_TEXTBOX_FUNC)
        self.txt_func = TextBox(ax_func, "f: ", initial="(x**2 - 1)/(x-1)")
        self.txt_func.on_submit(self._on_function_submit)
        
        # TextBox ponto
        ax_point = plt.axes(config.POS_TEXTBOX_POINT)
        self.txt_point = TextBox(ax_point, "p: ", initial="")
        self.txt_point.on_submit(self._on_point_submit)
        
        # Slider Z
        ax_slider = plt.axes(config.POS_SLIDER_Z)
        self.slider_z = Slider(ax_slider, "z", -3, 3, valinit=1.0)
        self.slider_z.on_changed(self._on_z_change)
        
        # Botão animar
        ax_btn = plt.axes(config.POS_BUTTON_ANIMATE)
        self.btn_animate = Button(ax_btn, "Animar")
        self.btn_animate.on_clicked(self._on_animate_click)

        # Checkbox para modos de animação (Auto, Zoom, Trace, Hold, Pulse)
        ax_check = plt.axes(config.POS_CHECK_ANIMATE)
        mode_labels = ['Auto', 'Zoom', 'Trace', 'Hold', 'Pulse']
        mode_actives = [False, True, True, True, False]
        self.check_modes = CheckButtons(ax_check, mode_labels, mode_actives)
        for lab, active in zip(mode_labels, mode_actives):
            if active and lab != 'Auto':    
                self.animation_modes.add(lab)
        self.check_modes.on_clicked(self._on_toggle_mode)
    
    def _plot_function(self, func_str: str):
        """Plota função principal"""
        try:
            # Parse função
            f = self.math_engine.parse_function(func_str)
            vars_ = sorted(f.free_symbols, key=lambda s: s.name)
            dimension = len(vars_)
            
            self.current_func = f
            self.current_vars = vars_
            
            # Recria subplot se necessário
            plot_dim = 1 if dimension <= 1 else 2
            if self.current_dimension != plot_dim:
                self.fig.delaxes(self.ax)
                if plot_dim == 1:
                    self.ax = self.fig.add_subplot(111)
                else:
                    self.ax = self.fig.add_subplot(111, projection='3d')
                self.current_dimension = plot_dim
                self.current_dimension = plot_dim
                # Para animator antigo antes de criar um novo (se houver)
                try:
                    if getattr(self, 'animator', None) is not None:
                        self.animator.stop()
                except Exception:
                    pass
                self.renderer = PlotRenderer(self.ax, self.fig)
                self.animator = LimitAnimator(self.ax, self.fig)
                self.fig.canvas.mpl_connect('button_press_event', self._on_click)
            
            # Limpa tela
            self.ax.clear()
            self.renderer.clear_artists()
            self.info_panel.clear()
            
            # Plota conforme dimensão
            if dimension == 0:
                # Função constante
                const_val = float(f)
                self.renderer.plot_constant_function(const_val, f)
            elif dimension == 1:
                # Função 1D
                f_num = self.math_engine.make_numeric(f, vars_)
                self.renderer.plot_1d_function(f_num, f)
            elif dimension == 2:
                # Função 2D
                f_num = self.math_engine.make_numeric(f, vars_)
                self.renderer.plot_2d_surface(f_num, f)
            elif dimension == 3:
                # Função 3D (fatia)
                f_num = self.math_engine.make_numeric(f, vars_)
                self.renderer.plot_3d_slice(f_num, f, self.z_value)
            else:
                self.ax.text(0.5, 0.5, "Dimensão não suportada",
                           ha='center', va='center', transform=self.ax.transAxes)
            
            # Desenha análise do ponto se existir
            if self.selected_point:
                self._analyze_point(self.selected_point)
            
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Erro ao plotar: {e}")
            self.ax.text(0.5, 0.5, f"Erro: {str(e)[:50]}",
                        ha='center', va='center', transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
    
    def _analyze_point(self, point: Tuple):
        """Analisa ponto selecionado"""
        if not self.current_func or not point:
            return
        
        dimension = len(self.current_vars)
        
        if dimension == 0:
            # Função constante
            x_val = point[0]
            const_val = float(self.current_func)
            self.renderer.draw_point_marker(x_val, const_val)
            self.info_panel.display_constant_info(x_val, const_val, self.current_func)
        
        elif dimension == 1:
            # Função 1D
            x_val = point[0]
            var = self.current_vars[0]
            
            # Calcula limites
            lim_right, exists_right = self.math_engine.calculate_limit(
                self.current_func, var, x_val, 'right')
            lim_left, exists_left = self.math_engine.calculate_limit(
                self.current_func, var, x_val, 'left')
            lim_global, exists_global = self.math_engine.calculate_limit(
                self.current_func, var, x_val, 'both')
            
            limits = {
                'right': lim_right,
                'left': lim_left,
                'global': lim_global
            }
            
            # Calcula derivada
            derivative = self.math_engine.calculate_derivative(
                self.current_func, self.current_vars, point)
            
            # Desenha marcadores
            f_val = self.math_engine.evaluate_at_point(
                self.current_func, self.current_vars, point)
            
            if f_val is not None and exists_global and isinstance(lim_global, (int, float)):
                filled = np.isclose(f_val, lim_global)
                self.renderer.draw_point_marker(x_val, lim_global, filled=filled)
            
            if f_val is None and exists_global and isinstance(lim_global, (int, float)):
                self.renderer.draw_point_marker(x_val, lim_global, filled=False)

            # Limites laterais (se global não existe)
            if not (exists_global and isinstance(lim_global, (int, float))):
                eps = 0.01
                if exists_right and isinstance(lim_right, (int, float)) and np.isfinite(lim_right):
                    self.renderer.draw_point_marker(x_val + eps, lim_right, 
                                                   filled=False, 
                                                   color=config.COLOR_LIMIT_RIGHT,
                                                   size=config.MARKER_SIZE_LATERAL)
                if exists_left and isinstance(lim_left, (int, float)) and np.isfinite(lim_left):
                    self.renderer.draw_point_marker(x_val - eps, lim_left,
                                                   filled=False,
                                                   color=config.COLOR_LIMIT_LEFT,
                                                   size=config.MARKER_SIZE_LATERAL)
            
            # Tangente
            _, df_val = derivative
            if df_val is not None and f_val is not None:
                self.renderer.draw_tangent_line(x_val, f_val, df_val)
            
            # Painel de informações
            self.info_panel.display_1d_info(x_val, limits, derivative)
            
            # Animação (se ativada)
            if self.animate_limits and not self.animator.is_running:
                self.animator.animate_limit_approach(
                    self.current_func, var, x_val, 'both')
        
        else:
            # Função 2D/3D
            x_val, y_val = point[0], point[1]
            z_val = point[2] if len(point) > 2 else self.z_value
            
            point_full = (x_val, y_val, z_val) if dimension == 3 else (x_val, y_val)
            
            # Calcula derivadas
            derivatives = self.math_engine.calculate_derivative(
                self.current_func, self.current_vars, point_full)
            
            # Desenha ponto
            f_val = self.math_engine.evaluate_at_point(
                self.current_func, self.current_vars, point_full)
            
            if f_val is not None:
                self.renderer.draw_3d_point(x_val, y_val, f_val)
                
                # Plano tangente
                _, deriv_vals = derivatives
                if deriv_vals and all(v is not None for v in deriv_vals):
                    self.renderer.draw_tangent_plane((x_val, y_val), f_val, deriv_vals)
            
            # Painel de informações
            self.info_panel.display_2d_info((x_val, y_val), derivatives)
        
        self.fig.canvas.draw_idle()
    
    def _on_click(self, event):
        """Callback de clique no gráfico"""
        if event.inaxes != self.ax:
            return
        
        dimension = len(self.current_vars)
        
        if dimension <= 1:
            self.selected_point = (event.xdata,)
        elif dimension in [2, 3]:
            if hasattr(event, 'xdata') and hasattr(event, 'ydata'):
                self.selected_point = (event.xdata, event.ydata)
                if dimension == 3:
                    self.selected_point = self.selected_point + (self.z_value,)
            else:
                return
        
        self._plot_function(self.txt_func.text)
    
    def _on_function_submit(self, text):
        """Callback de submissão de função"""
        self.selected_point = None
        self._plot_function(text)
    
    def _on_point_submit(self, text):
        """Callback de submissão de ponto"""
        try:
            if not text.strip():
                self.selected_point = None
                self._plot_function(self.txt_func.text)
                return
            
            # Remove formatação
            text = text.replace('(', '').replace(')', '').replace('p', '').replace('=', '').strip()
            coords = []
            
            # Parseia coordenadas (suporta expressões)
            for coord_str in text.split(','):
                expr = sp.sympify(coord_str.strip())
                coords.append(float(expr.evalf()))
            
            self.selected_point = tuple(coords)
            self._plot_function(self.txt_func.text)
        except Exception as e:
            print(f"Erro ao parsear ponto: {e}")
    
    def _on_z_change(self, val):
        """Callback de mudança no slider Z"""
        self.z_value = val
        if len(self.current_vars) == 3:
            self._plot_function(self.txt_func.text)
    
    def _on_animate_click(self, event):
        """Callback do botão de animar"""
        if self.selected_point and len(self.current_vars) == 1:
            x_val = self.selected_point[0]
            var = self.current_vars[0]
            try:
                self.animator.modes = set(self.animation_modes)
            except Exception:
                pass
            self.animator.animate_limit_approach(self.current_func, var, x_val, 'both')
    
    def _on_toggle_mode(self, label: str) -> None:
        """Callback para toggles de modos de animação. 'Auto' controla animação automática; os demais são efeitos."""
        if label == 'Auto':
            self.animate_limits = not self.animate_limits
            return
        if label in self.animation_modes:
            self.animation_modes.remove(label)
        else:
            self.animation_modes.add(label)
        try:
            if getattr(self, 'animator', None) is not None:
                self.animator.modes = set(self.animation_modes)
        except Exception:
            pass

    def _on_toggle_animate(self, label: str) -> None:
        """Compatibilidade: toggle antigo"""
        self.animate_limits = not self.animate_limits
    
    def show(self):
        """Exibe a aplicação"""
        plt.show()

# MAIN

if __name__ == "__main__":
    app = FunctionPlotter()
    app.show()