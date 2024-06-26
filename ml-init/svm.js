let ml = require('machine_learning');
let x = [[0.4, 0.5, 0.5, 0, 0, 0.],
[0.5, 0.3, 0.5, 0, 0, 0.01],
[0.4, 0.8, 0.5, 0, 0.1, 0.2],
[1.4, 0.5, 0.5, 0, 0, 0.],
[1.5, 0.3, 0.5, 0, 0, 0.],
[0, 0.9, 1.5, 0, 0, 0.],
[0, 0.7, 1.5, 0, 0, 0.],
[0.5, 0.1, 0.9, 0, -1.8, 0.],
[0.8, 0.8, 0.5, 0, 0, 0.],
[0, 0.9, 0.5, 0.3, 0.5, 0.2],
[0, 0, 0.5, 0.4, 0.5, 0.],
[0, 0, 0.5, 0.5, 0.5, 0.],
[0.3, 0.6, 0.7, 1.7, 1.3, -0.7],
[0, 0, 0.5, 0.3, 0.5, 0.2],
[0, 0, 0.5, 0.4, 0.5, 0.1],
[0, 0, 0.5, 0.5, 0.5, 0.01],
[0.2, 0.01, 0.5, 0, 0, 0.9],
[0, 0, 0.5, 0.3, 0.5, -2.3],
[0, 0, 0.5, 0.4, 0.5, 4],
[0, 0, 0.5, 0.5, 0.5, -2]];

let y = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

let svm = new ml.SVM({
    x: x,
    y: y
});

svm.train({
    C: 1.1,
    tol: 1e-5,
    max_passes: 50,
    alpha_tol: 1e-5,

    kernel: { type: "polynomial", c: 2, d: 6 }
});

console.log("Natija : ", svm.predict([1.3, 1.7, 0.5, 0.5, 1.5, 0.4]));