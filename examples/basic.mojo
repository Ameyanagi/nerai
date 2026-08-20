from nerai import LossKind, TerminationReason, evaluate_loss, robust_cost


def main() raises:
    var evaluation = evaluate_loss(LossKind.HUBER, 4.0)
    var cost = robust_cost(LossKind.HUBER, 4.0, scale=2.0)

    print("Huber rho(4):", evaluation.value)
    print("Huber cost for residual=4, scale=2:", cost)
    print(
        "Gradient tolerance means convergence:",
        TerminationReason.GRADIENT_TOLERANCE.is_success(),
    )
