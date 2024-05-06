let ml = require('machine_learning');

let data = [[1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
[1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
[1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
[0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
];

let result = [23, 12, 23, 23, 45, 70, 123, 73, 146, 158, 64];

let knn = new ml.KNN({
    data: data,
    result: result
});

let y = knn.predict({
    x: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    k: 2,

    weightf: { type: 'gaussian', sigma: 10.0 },

    distance: { type: 'euclidean' }
});

console.log(y);