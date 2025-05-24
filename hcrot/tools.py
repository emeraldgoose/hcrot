import numpy as np

def finite_difference_grad_check(model, input_sample, timesteps, epsilon=1e-5, atol=1e-4, rtol=1e-2):
    # Step 1: Forward pass
    output = model(input_sample, timesteps)
    loss = np.sum(output)  # dummy loss: sum of all outputs

    # Step 2: Backward pass
    doutput = np.ones_like(output)
    _, dw_analytic, db_analytic = model.backward(doutput)

    # Step 3: Finite difference gradient estimation
    for name, param in model.parameters.items():
        # Fix: Check explicitly if the key exists
        if name in dw_analytic:
            grad_analytic = dw_analytic[name]
        elif name in db_analytic:
            grad_analytic = db_analytic[name]
        else:
            continue

        grad_numeric = np.zeros_like(param)

        # Iterate over each element of the parameter
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index

            # perturb positively
            param[idx] += epsilon
            pos_output = model(input_sample, timesteps)
            pos_loss = np.sum(pos_output)

            # perturb negatively
            param[idx] -= 2 * epsilon
            neg_output = model(input_sample, timesteps)
            neg_loss = np.sum(neg_output)

            # restore original value
            param[idx] += epsilon

            # numerical gradient
            grad_numeric[idx] = (pos_loss - neg_loss) / (2 * epsilon)

            it.iternext()

        # Step 4: Compare gradients
        diff = np.abs(grad_numeric - grad_analytic)
        rel_error = diff / (np.abs(grad_numeric) + np.abs(grad_analytic) + 1e-8)

        max_abs = np.max(diff)
        max_rel = np.max(rel_error)

        print(f"[{name}] max_abs_err = {max_abs:.6e}, max_rel_err = {max_rel:.6e}")

        if not (max_abs < atol or max_rel < rtol):
            print(f"❌ Gradient check failed for parameter: {name}")
        else:
            print(f"✅ Gradient check passed for parameter: {name}")