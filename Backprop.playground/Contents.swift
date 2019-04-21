import Foundation

// Training hyperparams
let 🕰: Int = 10
let η: Double = 0.000_1

// Data
var train: (x: [[Double]], y: [Double])
train.x = [[1,4,1],[1,2,8],[1,1,0],[1,3,2],[1,1,4],[1,6,7]]
train.y = [2,-14,1,-1,-7,-8]
var w: [Double] = [0.0, 0.005, 0.07]
var ŷ: Double = 0.0
var L: Double = 0.0

// Gradients
var 𝞉L𝞉w: [Double] = [0, 0, 0]
var 𝞉ŷ𝞉w: [Double] = [1, 0, 0]
var 𝞉L𝞉ŷ: Double = 0.0

func shuffleData(_ data: inout (x: [[Double]], y: [Double])) {
    let shuffledInd = data.x.indices.shuffled()
    data.x = shuffledInd.map { data.x[$0] }
    data.y = shuffledInd.map { data.y[$0] }
}

func model(placeholder x: [Double], weights w: [Double]) -> Double {
    let y = (zip(w, x).map(*)).reduce(0, +)
    print("ŷ: \(y)")
    return y
}

func lossFunction(_ a: Double, _ b: Double) -> Double {
    let loss = 0.5 * pow(a - b, 2)
    print("Loss: \(round(loss * 100) / 100)")
    return loss
}

func updateGradients(given batch: (x: [Double], y: Double)) {
    𝞉L𝞉ŷ = batch.y - ŷ
    𝞉ŷ𝞉w[1...] = batch.x[1...]
    𝞉L𝞉w = 𝞉ŷ𝞉w.map { 𝞉L𝞉ŷ * $0 }
    
    print("𝞉L𝞉ŷ: \(𝞉L𝞉ŷ)")
    print("𝞉ŷ𝞉w: \(𝞉ŷ𝞉w)")
    print("𝞉L𝞉w: \(𝞉L𝞉w)")
}

func updateWeights() {
    print("w: \(w)")
    w = zip(w,𝞉L𝞉w).map { $0 - η * $1 }
}

// Actual code
for _ in 0..<🕰 {
    shuffleData(&train)
    for (currentX,currentY) in zip(train.x,train.y) {
        print("x: \(currentX)")
        print("y: \(currentY)")
        ŷ = model(placeholder: currentX, weights: w)
        L = lossFunction(currentY, ŷ)
        updateGradients(given: (currentX,currentY))
        updateWeights()
    }
    print()
}

// Final
print("y = \(w[1])x1 + \(w[2])x2 + \(w[0])")
