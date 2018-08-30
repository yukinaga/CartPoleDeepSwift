//
//  GameScene.swift
//  CertPole
//
//  Created by Yukinaga2 on 2018/08/13.
//  Copyright © 2018年 Yukinaga Azuma. All rights reserved.
//

import SpriteKit
import GameplayKit

class GameScene: SKScene{
    
    let maxStep = 200
    let maxEpisode = 1000
    let moveDistance:CGFloat = 5
    
    var episode = 0
    var step = 0
    
    var pole : SKShapeNode!
    var cart : SKSpriteNode!
    var poleSize: CGSize!
    var cartSize: CGSize!
    
    let agent = Agent()
    
    var poleAngleRecord = [CGFloat]()
    var poleVelocityRecord = [CGFloat]()
    
    override func didMove(to view: SKView) {
        
        cartSize = CGSize(width: 100, height: 100)

        cart = SKSpriteNode(imageNamed: "robot_normal.png")
        cart.size = cartSize
        cart.physicsBody = SKPhysicsBody(rectangleOf: cartSize)
        cart.physicsBody?.affectedByGravity = false
        cart.physicsBody?.isDynamic = false
        
        poleSize = CGSize(width: 10, height: 200)

        pole = SKShapeNode(rectOf: poleSize)
        pole.fillColor = UIColor.yellow
        pole.physicsBody = SKPhysicsBody(rectangleOf: poleSize)
        pole.physicsBody?.friction = 0
        
        setInitialState()
        
        let joint = SKPhysicsJointPin.joint(withBodyA: pole.physicsBody!, bodyB: cart.physicsBody!, anchor: CGPoint(x: 0, y: -100))
        
        self.addChild(pole)
        self.addChild(cart)
        
        self.physicsWorld.add(joint)
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
//        let randomForce = CGFloat(Int(arc4random_uniform(21))-10)
//        cart.physicsBody?.applyForce(CGVector(dx: force, dy: 0))
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {        
        let touch = touches.first
        let dx = (touch?.location(in: self).x)! - (touch?.previousLocation(in: self).x)!
        cart.position = CGPoint(x: cart.position.x+dx, y: cart.position.y)
    }
    
    override func update(_ currentTime: TimeInterval) {
        
//        recordStatus()
        
        // Update Q table
        if pole.zRotation < -CGFloat.pi/4 || pole.zRotation > CGFloat.pi/4 {
            agent.updateQNetwork(reward: -1, nextState: getState())
            cart.texture = SKTexture(imageNamed: "robot_sad.png")
            refreshEpisode()
            return
        }else if step == 0 {
            
        }else if step >= maxStep {
            agent.updateQNetwork(reward: 1, nextState: getState())
            cart.texture = SKTexture(imageNamed: "robot_pleasure.png")
            refreshEpisode()
            return
        }else{
            agent.updateQNetwork(reward: 0, nextState: getState())
        }
        
        // Action
        let action = agent.getAction(currenState: getState(), episode: episode)
        let distance = action==0 ? -moveDistance : moveDistance
        cart.position = CGPoint(x: cart.position.x+distance, y: cart.position.y)
        
        step += 1
    }
    
    func setInitialState(){
        cart.position = CGPoint(x:self.frame.midX, y:-150)
        cart.physicsBody?.velocity = CGVector(dx: 0, dy: 0)
        
        pole.position = CGPoint(x:0, y:0)
        pole.zRotation = 0
        pole.physicsBody?.velocity = CGVector(dx: 0, dy: 0)
        pole.physicsBody?.angularVelocity = 0
    }
    
    func refreshEpisode(){
        step = 0
        episode += 1
        let episodeLabel = self.childNode(withName: "episodeLabel") as! SKLabelNode
        episodeLabel.text = "Episode = " + String(episode)
        if episode >= maxEpisode {
            self.isPaused = true
        }
        setInitialState()
    }
    
    func getState() -> [String:CGFloat]{
        return ["pole_angle":pole.zRotation, "pole_speed":(pole.physicsBody?.angularVelocity)!]
    }
    
    func recordStatus(){
        let randomForce = CGFloat(Int(arc4random_uniform(101))-50)
        pole.physicsBody?.applyForce(CGVector(dx: randomForce, dy: 0))
        
        if pole.zRotation < -CGFloat.pi/4 || pole.zRotation > CGFloat.pi/4 {
            setInitialState()
            episode += 1
        }
            
        poleAngleRecord.append(pole.zRotation)
        poleVelocityRecord.append((pole.physicsBody?.angularVelocity)!)
        
        if episode >= 100 {
            self.isPaused = true
            print(getStd(array: poleVelocityRecord))
            print(getMaxMin(array: poleVelocityRecord))
            print(poleVelocityRecord)
        }
    }
    
    func getStd(array:[CGFloat]) -> CGFloat{
        var mu:CGFloat = 0
        for v in array {
            mu += v
        }
        mu /= CGFloat(array.count)
        
        var std:CGFloat = 0
        for v in array {
            std += (v-mu)*(v-mu)
        }
        std /= CGFloat(array.count)
        
        return sqrt(std)
    }
    
    func getMaxMin(array:[CGFloat]) -> [String:CGFloat] {
        var max:CGFloat = array[0]
        var min:CGFloat = array[0]
        for v in array {
            max = v > max ? v : max
            min = v < min ? v : min
        }
        return ["Max":max, "Min":min]
    }
}
