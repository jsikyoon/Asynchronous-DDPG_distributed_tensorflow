# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os
import time
import sys
import argparse
import filter_env
from gym import wrappers
from ddpg import *
import gc
gc.enable()

FLAGS=None;
ENV_NAME = 'Reacher-v1'
EPISODES = 100000
local_step=1
TEST=10

def train():
  # parameter server and worker information
  ps_hosts = np.zeros(FLAGS.ps_hosts_num,dtype=object);
  worker_hosts = np.zeros(FLAGS.worker_hosts_num,dtype=object);
  port_num=FLAGS.st_port_num;
  for i in range(FLAGS.ps_hosts_num):
    ps_hosts[i]=str(FLAGS.hostname)+":"+str(port_num);
    port_num+=1;
  for i in range(FLAGS.worker_hosts_num):
    worker_hosts[i]=str(FLAGS.hostname)+":"+str(port_num);
    port_num+=1;
  ps_hosts=list(ps_hosts);
  worker_hosts=list(worker_hosts);
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join();
  elif FLAGS.job_name == "worker":
    device=tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" % FLAGS.task_index,
          cluster=cluster);

    #tf.set_random_seed(1);
    # env and model call
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env,device)

    # prepare session
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
      global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False);
      global_step_ph=tf.placeholder(global_step.dtype,shape=global_step.get_shape());
      global_step_ops=global_step.assign(global_step_ph);
      score = tf.get_variable('score',[],initializer=tf.constant_initializer(-21),trainable=False);
      score_ph=tf.placeholder(score.dtype,shape=score.get_shape());
      score_ops=score.assign(score_ph);
      init_op=tf.global_variables_initializer();
      # summary for tensorboard
      tf.summary.scalar("score", score);
      summary_op = tf.summary.merge_all()
      saver = tf.train.Saver();
    
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                   global_step=global_step,
                                   logdir=FLAGS.log_dir,
                                   summary_op=summary_op,
                                   saver=saver,
                                   init_op=init_op)
    
    with sv.managed_session(server.target) as sess:
      agent.set_sess(sess);
      while True:
        if sess.run([global_step])[0] > EPISODES:
          break
        score=0;
        for ls in range(local_step):
          state = env.reset();
          for step in xrange(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
              break;
        for i in xrange(TEST):
          state = env.reset()
          for j in xrange(env.spec.timestep_limit):
            #env.render()
            action = agent.action(state) # direct action for test
            state,reward,done,_ = env.step(action)
            score += reward
            if done:
              break
        sess.run(global_step_ops,{global_step_ph:sess.run([global_step])[0]+local_step});
        sess.run(score_ops,{score_ph:score/TEST/200});
        print(str(FLAGS.task_index)+","+str(sess.run([global_step])[0])+","+str(score/TEST/200));
    sv.stop();
    print("Done");

def main(_):
  #os.system("rm -rf "+FLAGS.log_dir);
  FLAGS.ps_hosts_num+=1;
  FLAGS.worker_hosts_num+=1;
  train()
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts_num",
      type=int,
      default=5,
      help="The Number of Parameter Servers"
  )
  parser.add_argument(
      "--worker_hosts_num",
      type=int,
      default=10,
      help="The Number of Workers"
  )
  parser.add_argument(
      "--hostname",
      type=str,
      default="jaesik-System-Product-Name",
      help="The Hostname of the machine"
  )
  parser.add_argument(
      "--st_port_num",
      type=int,
      default=2222,
      help="The start port number of ps and worker servers"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  # Log folder 
  parser.add_argument(
      "--log_dir",
      type=str,
      default="/tmp/addpg_log/",
      help="log folder name"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
