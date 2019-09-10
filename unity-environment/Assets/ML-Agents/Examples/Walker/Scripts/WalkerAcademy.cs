using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using System.IO;

public class WalkerAcademy : Academy
{
    public class HumanoidState
    {
        public HumanoidState(int boneCount)
        {
            pos = new Vector3[boneCount];
            rot = new Quaternion[boneCount];
            vel = new Vector3[boneCount];
            aVel = new Vector3[boneCount];

            touchingGround = new bool[boneCount];
            touchingWall = new bool[boneCount];
            touchingHold = new bool[boneCount];
            touchingHoldId = new int[boneCount];
            touchingTarget = new bool[boneCount];
        }

        // these variables comes from bones
        public Vector3[] pos;
        public Quaternion[] rot;
        public Vector3[] vel;
        public Vector3[] aVel;

        public bool[] touchingGround;
        public bool[] touchingWall;
        public bool[] touchingHold;
        public int[] touchingHoldId;
        public bool[] touchingTarget;

        public float current_accumulative_reward = 0f;
        public int current_step = 0;
    };

    public class WalkerTrajectory
    {
        public List<int> agent_state_ids = new List<int>();
        public float trajectory_value = 0.0f;

        public int count_valid_states = 0;
    };

    public int MaxBufferSize = 20480;
    public float Threshold = 3.0f;
    int currentSampleNum = 0;

    const string base_file_name = "statistics_walker_psi_";

    const bool flag_reset_trajectories = true;

    [HideInInspector] const int MaxTrajectoryStoredNumber = 2000;

    [HideInInspector] public List<HumanoidState> savedStates = new List<HumanoidState>();
    [HideInInspector] public List<WalkerTrajectory> savedTrajectories = new List<WalkerTrajectory>();
    [HideInInspector] public List<int> sortedTrajectoriesIndices = new List<int>();

    StreamWriter streamWriter = null;
    int nonForkedEpisodeCounter = 0;
    float nonForked_accumulative_reward = 0.0f;
    int loggedStep = 0;
    int maxStartingStep = 0;

    public int GetNextFreeState(int boneCount)
    {
        savedStates.Add(new HumanoidState(boneCount));

        return savedStates.Count - 1;
    }

    public int CopyState(int _fromStateId)
    {
        if (_fromStateId < 0)
            return -1;

        int nStateID = GetNextFreeState(savedStates[_fromStateId].pos.Length);

        CopyState(_fromStateId, nStateID);

        return nStateID;
    }

    public int CopyState(int _fromStateId, int _toStateId)
    {
        HumanoidState _fState = savedStates[_fromStateId];

        int boneCount = _fState.pos.Length;
        int nStateID = _toStateId;

        HumanoidState _cState = savedStates[nStateID];
        _cState.current_accumulative_reward = _fState.current_accumulative_reward;
        _cState.current_step = _fState.current_step;

        for (int i = 0; i < boneCount; i++)
        {
            _cState.pos[i] = _fState.pos[i];
            _cState.rot[i] = _fState.rot[i];
            _cState.vel[i] = _fState.vel[i];
            _cState.aVel[i] = _fState.aVel[i];

            _cState.touchingGround[i] = _fState.touchingGround[i];
            _cState.touchingWall[i] = _fState.touchingWall[i];

            _cState.touchingHold[i] = _fState.touchingHold[i];
            _cState.touchingHoldId[i] = _fState.touchingHoldId[i];

            _cState.touchingTarget[i] = _fState.touchingTarget[i];
        }

        return nStateID;
    }

    public void UpdateTrajectory(int updatingTrajectoryIdx, int slotIndex, float current_accumulative_reward)
    {
        if (flag_reset_trajectories || updatingTrajectoryIdx < 0 || updatingTrajectoryIdx >= savedTrajectories.Count || slotIndex < 0)
            return;

        int updating_step = savedStates[slotIndex].current_step;
        if (updating_step < savedTrajectories[updatingTrajectoryIdx].agent_state_ids.Count)
        {
            float c_val = savedStates[savedTrajectories[updatingTrajectoryIdx].agent_state_ids[updating_step]].current_accumulative_reward;

            if (c_val < current_accumulative_reward)
            {
                CopyState(slotIndex, savedTrajectories[updatingTrajectoryIdx].agent_state_ids[updating_step]);
            }
        }
        return;
    }

    public void IncreaseSampleCount()
    {
        currentSampleNum++;
    }

    void ResetTrajecotriesIfNecessary()
    {
        if (File.Exists(@"C:\Kourosh\Project\ml-agents-0.4.0a\python\update_frequency.txt"))
        {
            StreamReader streamReader = new StreamReader(@"C:\Kourosh\Project\ml-agents-0.4.0a\python\update_frequency.txt");
            int stepUpdate = int.Parse(streamReader.ReadLine());
            streamReader.Close();

            File.Delete(@"C:\Kourosh\Project\ml-agents-0.4.0a\python\update_frequency.txt");

            int cStep = stepUpdate;//
            if (cStep != loggedStep)
            {
                WriteStatistics(cStep);

                currentSampleNum = 0;
                if (flag_reset_trajectories)
                {
                    sortedTrajectoriesIndices.Clear();
                    maxStartingStep = 0;
                }
            }
        }
    }
    
    public void AddTrajectory(List<int> agent_state_ids, int current_step, int starting_step, bool isTrajectoryForked)
    {
        if (current_step >= agent_state_ids.Count)
        {
            current_step = agent_state_ids.Count - 1;
        }

        if (current_step > 0 && agent_state_ids[current_step] > 0)
        {
            float trajectory_value = savedStates[agent_state_ids[current_step]].current_accumulative_reward;
            if (!isTrajectoryForked)
            {
                nonForked_accumulative_reward += trajectory_value;
                nonForkedEpisodeCounter++;
            }
        }

        if (starting_step > maxStartingStep)
        {
            return;
        }

        maxStartingStep = Mathf.Max(maxStartingStep, current_step);

        if (current_step > 0)
        {
            float trajectory_value = savedStates[agent_state_ids[current_step]].current_accumulative_reward;

            int insert_index = -1;
            for (int t = 0; t < sortedTrajectoriesIndices.Count && insert_index < 0; t++)
            {
                if (savedTrajectories[sortedTrajectoriesIndices[t]].trajectory_value < trajectory_value)
                {
                    insert_index = t;
                }
            }
            if (insert_index < 0)
            {
                if (sortedTrajectoriesIndices.Count == 0)
                    insert_index = 0;
                else
                {
                    insert_index = sortedTrajectoriesIndices.Count;
                }
            }

            if (insert_index == MaxTrajectoryStoredNumber)
                insert_index--;

            if (savedTrajectories.Count < MaxTrajectoryStoredNumber && sortedTrajectoriesIndices.Count == savedTrajectories.Count)
            {
                // here we can add new trajectory
                savedTrajectories.Add(new WalkerTrajectory());
                int nIndex = savedTrajectories.Count - 1;

                for (int i = 0; i < current_step + 1; i++)
                {
                    int nStateID = CopyState(agent_state_ids[i]);
                    if (nStateID > -1)
                    {
                        savedTrajectories[nIndex].agent_state_ids.Add(nStateID);
                    }
                }
                savedTrajectories[nIndex].trajectory_value = trajectory_value;
                savedTrajectories[nIndex].count_valid_states = savedTrajectories[nIndex].agent_state_ids.Count;

                sortedTrajectoriesIndices.Insert(insert_index, nIndex);
            }
            else
            {
                int nIndex = -1;
                if (sortedTrajectoriesIndices.Count < savedTrajectories.Count)
                {
                    nIndex = sortedTrajectoriesIndices.Count;
                }
                else
                {
                    // here we need to swap indices (remove last, update, re-insert)
                    nIndex = sortedTrajectoriesIndices[sortedTrajectoriesIndices.Count - 1];

                    sortedTrajectoriesIndices.Remove(nIndex);
                }
                int counter = 0;
                for (int i = 0; i < current_step + 1; i++)
                {
                    if (counter < savedTrajectories[nIndex].agent_state_ids.Count)
                    {
                        int nStateID = savedTrajectories[nIndex].agent_state_ids[counter];
                        CopyState(agent_state_ids[i], nStateID);
                    }
                    else
                    {
                        int nStateID = CopyState(agent_state_ids[i]);
                        savedTrajectories[nIndex].agent_state_ids.Add(nStateID);
                    }
                    counter++;
                }
                savedTrajectories[nIndex].trajectory_value = trajectory_value;
                savedTrajectories[nIndex].count_valid_states = counter;

                sortedTrajectoriesIndices.Insert(insert_index, nIndex);
            }
        }
        return;
    }

    private void OpenStatisticsFile()
    {
        if (streamWriter == null)
        {
            int i = 0;

            while (File.Exists(base_file_name + i.ToString() + ".txt"))
            {
                i++;
            } 

            string file_name = base_file_name + i.ToString() + ".txt";
            if (File.Exists(file_name))
                streamWriter = new StreamWriter(file_name, true);
            else
                streamWriter = new StreamWriter(file_name);
        }
    }

    private void WriteStatistics(int cStep)
    {
        if (nonForkedEpisodeCounter > 0)
        {
            Debug.Log(nonForkedEpisodeCounter.ToString() + "," + ((int)GetStepCount() / 5).ToString() + "," + cStep.ToString() + "," 
                + ((float)nonForked_accumulative_reward / (float)(nonForkedEpisodeCounter + 1e-6)).ToString());

            if (streamWriter == null)
            {
                OpenStatisticsFile();
            }
            streamWriter.WriteLine(cStep.ToString() + "," + ((float)nonForked_accumulative_reward / (float)(nonForkedEpisodeCounter + 1e-6)).ToString());
            streamWriter.Flush();

            nonForked_accumulative_reward = 0;
            nonForkedEpisodeCounter = 0;

            loggedStep = cStep;
        }
        return;
    }

    public override void InitializeAcademy()
    {
        Monitor.verticalOffset = 1f;

        // We increase the Physics solver iterations in order to
        // make walker joint calculations more accurate.
        Physics.defaultSolverIterations = 12;
        Physics.defaultSolverVelocityIterations = 12;
        Time.fixedDeltaTime = 0.01333f; //(75fps). default is .2 (60fps)
        Time.maximumDeltaTime = .15f; // Default is .33

        OpenStatisticsFile();
    }

    public override void AcademyReset()
    {
    }

    public override void AcademyStep()
    {
        Time.timeScale = trainingConfiguration.timeScale;

        ResetTrajecotriesIfNecessary();
    }

    private void OnApplicationQuit()
    {
        streamWriter.Close();
    }
}
