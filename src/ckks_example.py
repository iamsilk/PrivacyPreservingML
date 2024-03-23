from eva import *
poly = EvaProgram('Polynomial', vec_size=1024)
with poly:
    x = Input('x')
    Output('y', 3*x**2 + 5*x - 2)

poly.set_output_ranges(30)
poly.set_input_scales(30)

from eva.ckks import *
compiler = CKKSCompiler()
compiled_poly, params, signature = compiler.compile(poly)

print(compiled_poly.to_DOT())

from eva.seal import *
public_ctx, secret_ctx = generate_keys(params)

inputs = { 'x': [i for i in range(compiled_poly.vec_size)] }
encInputs = public_ctx.encrypt(inputs, signature)

encOutputs = public_ctx.execute(compiled_poly, encInputs)

outputs = secret_ctx.decrypt(encOutputs, signature)
print(outputs)

from eva.metric import valuation_mse
reference = evaluate(compiled_poly, inputs)
print('MSE', valuation_mse(outputs, reference))