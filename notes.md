

# rl tutorial
https://github.com/keras-rl/keras-rl

# other people's work
https://github.com/mws262/MATLAB-Reinforcement-Learning-Pendulum/blob/master/QlearnPend.m


# states=x,y,dtheta, for continuousty 
11.25：我用10×10的relu试了下； inverted_pend,500k收敛；down_pend,一晚上，跑不出来
11.26：我把力改成了2-->3，网络10*15*10
    inverted_pend, 80k就收敛的很好了
    down_pend,跑了30万吧，效果明显比网络没改前好。但还是感觉不够，我于是直接又加了一层。
11.26：网络10*15*15*10
    inverted_pend, 200k有收敛，但还是有一点点不稳定。30k够了。
    down_pend, train到800k已经能摆起来了，就差最后收敛了

    每个epoch的时间，我之前设了50秒，但发现有个问题。如果摆在40秒的时候时候到了顶上，那么在顶上就没待多久。哦，这应该不是个问题，毕竟数据都是存在memory里的。没问题，没问题。
    问题是：DQN里与没有个机制，在到了终点以后，把最后那部分，反复训练？
    
11.26周一晚上跑了一晚上，还是没收敛。
我把最大速度从5改成了4，把上层区间（上下层reward不一样）设为了 +-2/3*pi，再跑一下。还是不行。
我发现，如果用x,y,dq,就会把inv_pend给搞坏掉。所以还是用q,dq当成状态吧



