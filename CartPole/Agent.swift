//
//  Agent.swift
//  CertPole
//
//  Created by Yukinaga2 on 2018/08/16.
//  Copyright © 2018年 Yukinaga Azuma. All rights reserved.
//

import UIKit

class Agent {
    
    var qNetwork = NeuralNetwork()

    let digitizeNumber = 24
    let gamma = 0.99
    
    let limit = ["pole_angle":CGFloat.pi/4, "pole_speed":CGFloat(1.2)]
    var currentStateN = ["pole_angle":CGFloat(0), "pole_speed":CGFloat(0)]
    var action = 0
    
    init() {

    }
    
    func normalizeState(state:[String:CGFloat]) -> [String:CGFloat]{
        var normalizedState = [String:CGFloat]()
        for (key, _) in state {
            let normal = state[key]! / limit[key]!
            normalizedState[key] = normal
        }
        return normalizedState
    }
    
    func getAction(currenState:[String:CGFloat], episode:Int) -> Int{
        currentStateN = normalizeState(state: currenState)
        let epsilon = 0.5*(1.0/(CGFloat(episode)+1.0))
        
        let x = [Double(currentStateN["pole_angle"]!), Double(currentStateN["pole_speed"]!)]
        let y = qNetwork.predict(x: x, isNext: false)
        
        if epsilon <= CGFloat(arc4random_uniform(1001))/1000.0 {
            action = y[0] > y[1] ? 0 : 1
        }else{
            action = arc4random_uniform(2)==0 ? 0 : 1
        }
        
        return action
    }
    
    func updateQNetwork(reward:Double, nextState:[String:CGFloat]){
        let nextStateN = normalizeState(state: nextState)
        let x = [Double(nextStateN["pole_angle"]!), Double(nextStateN["pole_speed"]!)]
        
        let y = qNetwork.predict(x: x, isNext: true)
        let maxIndex = y.maxIndex()
        var t = [Double]()
        for (i, q) in y.enumerated() {
            t.append(i==maxIndex ? reward+gamma*q : qNetwork.outputLayer.y[i])
        }
        qNetwork.train(t: t)        
    }
}

extension Array where Element == Double {
    func maxIndex() -> Int {
        var maxIndex = 0
        var max = self[0]
        for i in 0..<self.count {
            if self[i] > max {
                maxIndex = i
                max = self[i]
            }
        }
        return maxIndex
    }
}
