from manim import *

class CreateCircle(Scene):
   def construct(self):
      circle = Circle()
      circle.set_fill(BLUE, opacity=0.5)
      self.play(Create(circle))

class ConstraintSolver(Scene):
   def construct(self):

      solve = MathTex(r"\text{solve}").shift(LEFT * 3)

      mlcp_mat = Matrix([[r"A\lambda = b"], [r"\lambda_{min} \leq \lambda \leq \lambda_{max}"]], left_bracket="(", right_bracket=")")

      A_big_expanded = MathTex("A = J(q) M(q)^{-1} J(q)^{T}")
      b_big_expanded = MathTex(r"b = -J(q) \dot{q} - g")

      mlcp = VGroup(solve, mlcp_mat)
      emphasis_rect = SurroundingRectangle(mlcp_mat.get_rows()[1])

      constraint = MathTex(r"\lambda_{min} \leq \lambda \leq \lambda_{max}")

      lin_eq = MathTex(r"\text{solve}(", "A", "\lambda", "=", "b", ")")

      pita = Tex("Pain in the ass")
      lambda_bounds_pita = MathTex(r"\lambda_{min} \text{ and } \lambda_{max} \text{ have a mixture of finite and infinite values}")

      A_big = Matrix([["a_{11}", "a_{12}", "\cdots", "a_{1m}"], ["a_{21}", "a_{22}", "\cdots", "a_{2m}"], [r"\vdots", r"\vdots", "\ddots", r"\vdots"], ["a_{m1}", "a_{m2}", "\cdots", "a_{mm}"]])
      lambda_big = Matrix([["\lambda_{1}"], ["\lambda_{2}"], [r"\vdots"], ["\lambda_{m}"]])
      b_big = Matrix([["b_{1}"], ["b_{2}"], [r"\vdots"], ["b_{m}"]])

      A_big.shift(LEFT * 2.5)
      lambda_big.shift(RIGHT * 1.5)
      b_big.shift(RIGHT * 3.5)

      lineq_group = VGroup(A_big, lambda_big, Tex(r"=").shift(RIGHT * 2.5), b_big)

      gauss_seidel = MathTex(r"\lambda = \text{GaussSeidel}(A, b)")
      aggravation = Tex("...or any of the billion other direct methods")

      projected_gauss_seidel = MathTex(r"\lambda = \text{ProjectedGaussSeidel}(A, b, \lambda_{0}, \lambda_{min}, \lambda_{max}, N_{max})")
      pgs_aggravation = Tex("Converges linearly")

      sequential_impulses = Tex("Sequential Impulses")
      lemke = Tex("Lemke's algorithm")

      linear_system = Tex("System of Linear Equations")
      mlcp_label = Tex("Mixed Linear Complementarity Problem")

      you_cant_just_do_this = MathTex(r"\lambda \neq \text{clamp}(\text{GaussSeidel}(A, b), \lambda_{min}, \lambda_{max})")

      self.play(FadeIn(mlcp))
      self.wait(3)
      self.play(mlcp.animate.shift(UP * 1), FadeIn(A_big_expanded.shift(DOWN * 0.5)), FadeIn(b_big_expanded.shift(DOWN * 1.5)))
      self.wait(3)
      self.play(mlcp.animate.shift(DOWN * 1), FadeOut(A_big_expanded), FadeOut(b_big_expanded))
      self.wait(3)
      self.play(Circumscribe(mlcp_mat.get_rows()[1]))
      self.play(FadeIn(emphasis_rect))
      self.wait(1)
      self.play(Write(pita.shift(DOWN * 1.5)))
      self.wait(1)
      self.play(LaggedStart(FadeOut(pita), FadeOut(emphasis_rect), TransformMatchingShapes(mlcp, lin_eq), lag_ratio=0.75, run_time=2))
      self.wait(3)
      self.play(ShrinkToCenter(lin_eq), GrowFromCenter(lineq_group))
      self.play(Write(linear_system.shift(UP * 2.5)))
      self.wait(3)
      self.play(lineq_group.animate.shift(RIGHT * 15), FadeIn(gauss_seidel))
      self.wait(1.5)
      self.play(Write(aggravation.shift(DOWN)))
      self.wait(1.5)
      self.play(FadeOut(gauss_seidel), FadeOut(aggravation), FadeIn(lineq_group.shift(LEFT * 15)))
      self.wait(0.5)
      self.play(FadeIn(constraint.shift(DOWN * 2.25), shift=UP))
      self.wait(3)
      self.play(FadeIn(lambda_bounds_pita.shift(DOWN * 3.0)))
      self.wait(3)
      self.play(FadeOut(linear_system), FadeOut(lambda_bounds_pita))
      self.play(Write(mlcp_label.shift(UP * 2.5)))
      self.wait(3)
      self.play(FadeOut(lineq_group), FadeOut(constraint), FadeIn(you_cant_just_do_this))
      self.wait(3)
      self.play(FadeOut(you_cant_just_do_this), FadeIn(projected_gauss_seidel))
      self.wait(1.5)
      self.play(Write(pgs_aggravation.shift(DOWN)))
      self.wait(3)
      self.play(FadeOut(pgs_aggravation), FadeOut(projected_gauss_seidel), FadeIn(mlcp))
      self.wait(3)
      self.play(FadeOut(mlcp), FadeOut(mlcp_label))
      self.wait(3)
