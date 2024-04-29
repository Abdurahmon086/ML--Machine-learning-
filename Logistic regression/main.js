function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

function logisticRegression(data, mainPoint) {
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
        const x = data[i][0];
        const y = data[i][1];

        const h = sigmoid(mainPoint[0] + mainPoint[1] * x);

        sum += y * Math.log(h) + (1 - y) * Math.log(1 - h);
    }
    return -sum / data.length;
}

function optimize(data, mainPoint, alpha, iterations) {
    const m = data.length;
    for (let iter = 0; iter < iterations; iter++) {
        let sum0 = 0;
        let sum1 = 0;
        for (let i = 0; i < m; i++) {
            const x = data[i][0];
            const y = data[i][1];
            const h = sigmoid(mainPoint[0] + mainPoint[1] * x);
            sum0 += (h - y);
            sum1 += (h - y) * x;
        }

        const delta0 = sum0 / m;
        const delta1 = sum1 / m;
        mainPoint[0] -= alpha * delta0;
        mainPoint[1] -= alpha * delta1;
    }
    return mainPoint;
}

const data = [
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 1],
    [5, 1],
    [6, 1],
    [7, 1],
    [8, 1],
    [9, 1],
    [10, 1],
    [11, 1],
    [12, 0],
    [13, 0],
    [14, 1],
];

let mainPoint = [0, 0];


const alpha = 0.01;
const iterations = 1000;

mainPoint = optimize(data, mainPoint, alpha, iterations);

console.log("Optimallashtirilgan mainPoint:", mainPoint);

const x_new = 15;
const h_new = sigmoid(mainPoint[0] + mainPoint[1] * x_new);
console.log("Yangi qiymat:", h_new);

