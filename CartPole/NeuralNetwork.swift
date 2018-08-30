//
//  NeuralNetwork.swift
//  CertPoleDeep
//
//  Created by Yukinaga2 on 2018/08/23.
//  Copyright © 2018年 Yukinaga Azuma. All rights reserved.
//

import UIKit
import Accelerate

class NeuralNetwork {
    
    let eta = 0.02
    let w_width = 1.0
    let n_m:UInt = 100
    
    var middleLayer1:MiddleLayer!
    var middleLayer2:MiddleLayer!
    var outputLayer:OutputLayer!
    
    init() {
        middleLayer1 = MiddleLayer(un: 2, n: n_m, width: w_width)
        middleLayer2 = MiddleLayer(un: n_m, n: n_m, width: w_width)
        outputLayer = OutputLayer(un: n_m, n: 2, width: w_width)
    }
    
    func predict(x:[Double], isNext:Bool) -> [Double]{
        let y_m1 = middleLayer1.forward(x: x, isNext: isNext)
        let y_m2 = middleLayer2.forward(x: y_m1, isNext: isNext)
        let y_out = outputLayer.forward(x: y_m2, isNext: isNext)
        return y_out
    }
    
    func train(t:[Double]) {
        let dx_out = outputLayer.backward(t: t)
        let dx_m2 = middleLayer2.backward(dy: dx_out)
        _ = middleLayer1.backward(dy: dx_m2)
        
        middleLayer1.update(eta: eta)
        middleLayer2.update(eta: eta)
        outputLayer.update(eta: eta)
    }
}

class BaseLayer{
    var w: la_object_t!
    var b: la_object_t!
    var dW: la_object_t!
    var db: la_object_t!
    
    var x: [Double]!
    var y: [Double]!
    var dx: [Double]!
    
    init(un:UInt, n:UInt, width:Double) {
        let w_arr = getRandArray(size: un*n, width: width)
        self.w = makeMatrix(array: w_arr, rows: un, cols: n)
        
        let b_arr = [Double](repeating: 0.0, count: Int(n))
        self.b = makeMatrix(array: b_arr, rows: 1, cols: n)
    }
    
    func update(eta:Double){
        let wd = eta * self.dW
        self.w = self.w - wd
        
        let bd = eta * self.db
        self.b = self.b - bd
    }
}

class MiddleLayer: BaseLayer{
    
    func forward(x: [Double], isNext:Bool) -> [Double] {
        let x_mat = makeMatrix(array: x, rows: 1, cols: UInt(x.count))
        let u_mat = x_mat * w + b
        let u = u_mat.toArray
        let y = sigmoid(u: u)

        if !isNext {
            self.x = x
            self.y = y
        }

        return y
    }
    
    func backward(dy: [Double]) -> [Double]{
        let delta = gradSigmoid(dy: dy, y: y)

        let x_mat = makeMatrix(array: self.x, rows: 1, cols: UInt(x.count))
        let delta_mat = makeMatrix(array: delta, rows: 1, cols: UInt(delta.count))
        
        self.dW = x_mat.trans * delta_mat
        self.db = delta_mat
        
        let dx_mat = delta_mat * self.w.trans
        self.dx = dx_mat.toArray
        
        return self.dx
    }
}

class OutputLayer: BaseLayer{
    
    func forward(x: [Double], isNext:Bool) -> [Double] {
        let x_mat = makeMatrix(array: x, rows: 1, cols: UInt(x.count))
        let u_mat = x_mat * w + b
        let u = u_mat.toArray
        let y = u
        
        if !isNext {
            self.x = x
            self.y = y
        }
        
        return y
    }
    
    func backward(t: [Double]) -> [Double] {
        let delta = self.y - t
        
        let x_mat = makeMatrix(array: self.x, rows: 1, cols: UInt(x.count))
        let delta_mat = makeMatrix(array: delta, rows: 1, cols: UInt(delta.count))
        
        self.dW = x_mat.trans * delta_mat
        self.db = delta_mat
        
        let dx_mat = delta_mat * self.w.trans
        self.dx = dx_mat.toArray
        
        return dx
    }
}

extension la_object_t {
    var rows: UInt {
        return UInt(la_matrix_rows(self))
    }
    
    var cols: UInt {
        return UInt(la_matrix_cols(self))
    }
    
    var toArray: [Double] {
        var arrayBuf = [Double](repeating: 0.0, count: Int(rows * cols))
        la_matrix_to_double_buffer(&arrayBuf, cols, self)
        return arrayBuf
    }
    
    var matrix: [[Double]] {
        var matrix = [[Double]]()
        for row in 1...rows {
            let firstCol = Int(cols * (row - 1))
            let lastCol = Int(cols * row - 1)
            let partCols = Array(toArray[firstCol...lastCol])
            matrix.append(partCols)
        }
        return matrix
    }
    
    var matrixDescription: String {
        return matrix.reduce("")
        {(acc, rowVals) in
            acc +
                rowVals.reduce(""){(ac, colVal) in ac + "\(colVal) "} +
            "\n"
        }
    }
    
    var trans : la_object_t {
        return la_transpose(self)
    }
}

fileprivate func makeMatrix(array:[Double], rows:UInt, cols:UInt) -> la_object_t{
    let mat = la_matrix_from_double_buffer(array, rows, cols, cols,
                                     la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
    return mat
}

fileprivate func +(left: la_object_t, right: la_object_t) -> la_object_t {
    return la_sum(left, right)
}

fileprivate func -(left: la_object_t, right: la_object_t) -> la_object_t {
    return la_difference(left, right)
}

fileprivate func -(left: [Double], right: [Double]) -> [Double] {
    var result = [Double]()
    for i in 0..<left.count {
        result.append(left[i] - right[i])
    }
    return result
}

public func *(left: Double, right: la_object_t) -> la_object_t {
    return la_scale_with_double(right, left)
}

public func *(left: la_object_t, right: la_object_t) -> la_object_t {
    return la_matrix_product(left, right)
}

public func sigmoid(u:[Double]) -> [Double]{
    var y = [Double]()
    for v in u {
        y.append(1.0/(1+exp(-v)))
    }
    return y
}

public func gradSigmoid(dy:[Double], y:[Double]) -> [Double]{
    var delta = [Double]()
    for i in 0..<dy.count {
        delta.append(dy[i] * y[i] * (1.0 - y[i]))
    }
    return delta
}

public func relu(u:[Double]) -> [Double]{
    var y = [Double]()
    for v in u {
        y.append(v < 0 ? 0 : v)
    }
    return y
}

public func gradRelu(dy:[Double], y:[Double]) -> [Double]{
    var delta = [Double]()
    for i in 0..<dy.count {
        delta.append(y[i] < 0 ? 0 : dy[i])
    }
    return delta
}

fileprivate func getRandArray(size:UInt, width:Double) -> [Double]{
    var randArray = [Double]()
    for _ in 0..<size {
        let rand = (Double(arc4random_uniform(201)) - 100.0) / 100.0
        randArray.append(rand * width)
    }
    return randArray
}
