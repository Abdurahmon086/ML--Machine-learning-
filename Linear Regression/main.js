const x = [2, 4, 6, 8, 10, 12, 14];
const y = [8, 14, 20, 26, 32, 38, 44];

function linearRegression(x, y, learningRate, epochs) {
  let w = 0;
  let b = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    let yPredicted = x.map(xi => w * xi + b);
    let errors = yPredicted.map((yp, i) => yp - y[i]);

    let gradientW = (2 / x.length) * x.reduce((sum, xi, i) => sum + errors[i] * xi, 0);
    let gradientB = (2 / x.length) * errors.reduce((sum, err) => sum + err, 0);

    w -= learningRate * gradientW;
    b -= learningRate * gradientB;

    let loss = errors.reduce((sum, err) => sum + Math.pow(err, 2), 0) / x.length;

    console.log(`Epoch ${epoch + 1}, Loss: ${loss.toFixed(4)}, w: ${w.toFixed(4)}, b: ${b.toFixed(4)}`);
  }
  return { w, b };
}

const learningRate = 0.01;
const epochs = 1000;

const linearResult = linearRegression(x, y, learningRate, epochs);
console.log("Chiziqli regressiya natijasi:", linearResult);

function predictLinear(x) {
  return linearResult.w * x + linearResult.b;
}

console.log("Bashorat qiymati:", predictLinear(14));
