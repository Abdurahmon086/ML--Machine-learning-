function polynomialRegression(x, y, learningRate, epochs) {
    let w1 = 0;
    let w2 = 0;
    let b = 0;
  
    for (let epoch = 0; epoch < epochs; epoch++) {
      let yPredicted = x.map((xi) => w1 * Math.pow(xi, 2) + w2 * xi + b);
      let errors = yPredicted.map((yp, i) => yp - y[i]);
  
      let gradientW1 =
        (2 / x.length) *
        x.reduce((sum, xi, i) => sum + errors[i] * Math.pow(xi, 2), 0);
      let gradientW2 =
        (2 / x.length) * x.reduce((sum, xi, i) => sum + errors[i] * xi, 0);
      let gradientB = (2 / x.length) * errors.reduce((sum, err) => sum + err, 0);
  
      // Gradient descent
      w1 -= learningRate * gradientW1;
      w2 -= learningRate * gradientW2;
      b -= learningRate * gradientB;
  
  let loss =
        errors.reduce((sum, err) => sum + Math.pow(err, 2), 0) / x.length;
  
      console.log(
        `Epoch ${epoch + 1}, Loss: ${loss.toFixed(4)}, w1: ${w1.toFixed(
          4
        )}, w2: ${w2.toFixed(4)}, b: ${b.toFixed(4)}`
      );
  
      // Nan qiymatlarni tekshirish
      if (isNaN(w1) || isNaN(w2) || isNaN(b)) {
        console.log(
          `NaN qiymatlar chiqdi. Gradient descent to'g'rilanmagan. Epoch: ${
            epoch + 1
          }`
        );
        break;
      }
    }
    return { w1, w2, b };
  }
  // Ma'lumotlar
  const xValues = [1, 2, 3, 4, 5];
  const yValues = [9, 15, 23, 33, 45];
  // Hyperparameters
  const learningRate = 0.001;
  const epochs = 100000;
  // Regressiya natijasi
  const polyResult = polynomialRegression(xValues, yValues, learningRate, epochs);
  console.log("Ikkinchi darajali polynomial regressiya natijasi:", polyResult);
  function predictPoly(x) {
    return polyResult.w1 * Math.pow(x, 2) + polyResult.w2 * x + polyResult.b;
  }
  console.log("Bashorat qiymati:", predictPoly(5));
  