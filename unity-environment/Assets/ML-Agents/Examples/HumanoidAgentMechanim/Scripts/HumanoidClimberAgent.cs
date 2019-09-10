using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using System.IO;

public class HumanoidClimberAgent : Agent
{
    static int m_agent_counter = 0;
    public class EndBodyPart
    {
        public bool IsConnected
        {
            get { return isConnected; }
        }

        public EndBodyPart(Transform bodyPartInfo, HumanoidClimberAgent _p)
        {
            _bodyPartInfo = bodyPartInfo;

            _locker_body = _bodyPartInfo.GetComponent<Rigidbody>();

            _joint = _bodyPartInfo.GetComponent<ConfigurableJoint>();

            holdContact = _bodyPartInfo.GetComponent<HoldContact>();

            groundContact = _bodyPartInfo.GetComponent<GroundContact>();
            groundContact.agent = _p;

            wallContact = _bodyPartInfo.GetComponent<WallContact>();

            disconnectBodyPart();
        }

        public Vector3 getCurrentForce()
        {
            return _joint.currentForce;
        }

        public void connectBodyPart(float dis)
        {
            float maxDisToHolds = 0.5f;
            if (holdContact.touchingHold && holdContact.hold_id == target_hold_id && dis <= maxDisToHolds)
            {
                if (cHoldId != target_hold_id)
                {
                    _locker_body.isKinematic = true;

                    isConnected = true;
                    cHoldId = target_hold_id;
                }
            }
            else if ((cHoldId != target_hold_id && cHoldId >= 0) || (dis > maxDisToHolds && _locker_body.isKinematic))
            {
                disconnectBodyPart();
            }
        }

        public void disconnectBodyPart()
        {
            _locker_body.isKinematic = false;

            cHoldId = -1;
            isConnected = false;
        }

        public Rigidbody GetRigidBody()
        {
            return _locker_body;
        }

        public int cHoldId = -1;
        public int target_hold_id = -1;
        public Transform _bodyPartInfo;
        private Rigidbody _locker_body;
        private Joint _joint;
        public HoldContact holdContact;
        public GroundContact groundContact;
        public WallContact wallContact;
        bool isConnected = false;
    };

    [Header("Specific to Climber")]
    public HumanoidClimberAcademy HighLevelGuide;
    public int trajectoryNum = -1;

    [Header("Holds in Environment")]
    public Transform[] holdPoses;
    public Transform[] Obstacles;

    [Header("End Points Connectors")]
    public Transform RHConnector;
    public Transform LHConnector;
    public Transform RLConnector;
    public Transform LLConnector;

    [HideInInspector] public HumanoidMecanimRig jdController;
    bool isNewDecisionStep;
    int currentDecisionStep;
    List<EndBodyPart> connectBodyParts = new List<EndBodyPart>();
    float[] sampleVectorAction;
    float[] splineStoredVectorAction;
    
    //user control agent boolean params
    [HideInInspector] public const float localAgentScaling = 3.0f;
    [HideInInspector] public const bool useAgentInterface = false; // where the task instruction are coming from?
    [HideInInspector] public const bool reset_each_transition = true;
    [HideInInspector] public bool flag_random_shift = !useAgentInterface 
        && HumanoidClimberAcademy.current_training_type != HumanoidClimberAcademy.TrainingType.RandomTraining; // where the task instruction are coming from?
    [HideInInspector] const bool useSpline = false;
    [HideInInspector] const bool useObstacles = false;

    // for test performance
    Vector3 targetPlannedHoldPos = new Vector3(0f, 0f, 0f);
    
    // general variables for each task
    bool succeeded_transition = false;
    int task_planned_num = 0;
    int task_failed_num = 0;
    int task_counter_step = 0;
    bool isTaskCompleted = false;

    enum RewardType {DeepMimicDistance = 0, DeepMimicThreshold = 1, Percentage = 2 };
    const RewardType current_reward_type = RewardType.DeepMimicThreshold;

    // lader climbing
    enum LadderClimbingTaskType { GoUp = 0, GoRight = 1, GoLeft = 2, GoDown = 3 };
    LadderClimbingTaskType currentTastType = LadderClimbingTaskType.GoUp;
    List<int> not_moved_limbs = new List<int>();
    int current_idx_limb_planned = 0;

    //////////////////////////////////////begin for spline /////////////////////////////////////
    // if useSpline = true and useLinearSpline = false, then make observation size = 320, action size = 90 in unity
    // if useSpline = true and useLinearSpline = true, then make observation size = 320, action size = 45 in unity
    // if useSpline = false, then make observation size = 232, action size = 44 in unity
    const int nActionSize = 44;
    const bool useLinearSpline = false;
    const int nControlPoints = useLinearSpline ? 1 : 2;
    public List<RecursiveTCBSpline> spline = new List<RecursiveTCBSpline>();
    float[] interpolatedControl;
    int minSplineStep = 1, maxSplineStep = 5; // each spline step is 10 steps in action simulation
    //int max_spline_used = 1;
    float[] _sampledActionSpline;
    int num_spline_done = 0;
    int spline_steps_needed = 0;
    int counter_spline_not_updated = 0;
    List<Vector2> MinMaxControlValues = new List<Vector2>();
    ////////////////////////////////////////end for spline /////////////////////////////////////

    // init and target hold ids
    int[] init_hold_ids = { -1, -1, -1, -1 };
    int[] target_hold_ids = { -1, -1, -1, -1 };
    float[] current_dis_target = { float.MaxValue, float.MaxValue, float.MaxValue, float.MaxValue };
    float[] init_dis_target = { float.MaxValue, float.MaxValue, float.MaxValue, float.MaxValue };
    // starting stance id in the graph
    int fatherNodeID = -1;
    int graphActionIndex = -1;
    int counter_agent_running = 0;
    List<ulong> agent_tree_task = new List<ulong>();
    bool isTaskOnLadderClimbing = false;
    int current_count_limb_movment = 0;

    // saving agent's trajectory
    List<int> agent_state_ids = new List<int>();
    bool isTrajectoryForked = false;
    int current_step = 0; // index on the current valid saved state
    int starting_step = 0;
    float starting_accumulative_reward = 0;

    ///////////////////////////////////Start: agent interface////////////////////////////////////
    enum UserCommandType { UserForwardSimulate = 0, UserNone = 1 };

    public class UserTrajectoryPoints
    {
        public int cStateID = -1;
        public int[] target_ids = { -1, -1, -1, -1 };
    }

    Vector3 lookAtPos = new Vector3();
    Vector3 camera_pos = new Vector3(20f, -Mathf.PI / 2.0f, 0f);
    Vector3 lastMouseClicked = new Vector3();
    int current_select_state = 0;
    int selected_limb = -1;
    int selected_hold = -1;
    UserCommandType current_command_type = UserCommandType.UserNone;

    List<UserTrajectoryPoints> user_trajectory_points = new List<UserTrajectoryPoints>();

    void AddTrajectoryPoint()
    {
        user_trajectory_points.Add(new UserTrajectoryPoints());
        user_trajectory_points[user_trajectory_points.Count - 1].cStateID = HighLevelGuide.GetNextFreeState(jdController.getBonesCount(), spline.Count);

        SaveState(user_trajectory_points[user_trajectory_points.Count - 1].cStateID);

        for (int i = 0; i < 4; i++)
        {
            user_trajectory_points[user_trajectory_points.Count - 1].target_ids[i] = connectBodyParts[i].target_hold_id;
        }
    }

    void RemoveLastTrajectoryPoint()
    {
        current_command_type = UserCommandType.UserNone;
        if (isTaskCompleted)
        {
            if (user_trajectory_points.Count > 1)
            {
                user_trajectory_points.RemoveAt(user_trajectory_points.Count - 1);
            }
            else
            {
                for (int b = 0; b < 4; b++)
                    user_trajectory_points[0].target_ids[b] = -1;
            }
        }

        LoadState(user_trajectory_points[user_trajectory_points.Count - 1].cStateID);

        AgentReset();
        isTaskCompleted = true;
        return;
    }

    void SetColorToLimb(int limb_id, Color _color)
    {
        switch (limb_id)
        {
            case 0:
                LLConnector.gameObject.GetComponent<Renderer>().material.color = _color;
                //jdController.getBoneGameObject(HumanBodyBones.LeftFoot).GetComponent<Renderer>().material.color = _color; // footL.gameObject
                break;
            case 1:
                RLConnector.gameObject.GetComponent<Renderer>().material.color = _color;
                //jdController.getBoneGameObject(HumanBodyBones.RightFoot).GetComponent<Renderer>().material.color = _color; // footR.gameObject.
                break;
            case 2:
                LHConnector.gameObject.GetComponent<Renderer>().material.color = _color;
                //jdController.getBoneGameObject(HumanBodyBones.LeftHand).GetComponent<Renderer>().material.color = _color; // handL.gameObject.
                break;
            case 3:
                RHConnector.gameObject.GetComponent<Renderer>().material.color = _color;
                //jdController.getBoneGameObject(HumanBodyBones.RightHand).GetComponent<Renderer>().material.color = _color; // handR.gameObject.
                break;
        }
    }

    void SetColorToHolds(int hold_id, Color _color)
    {
        _color.a = 0.25f;
        if (hold_id > -1)
            holdPoses[hold_id].gameObject.GetComponent<Renderer>().material.color = _color;
        return;
    }

    bool flag_play_video = false;
    List<int> path_index_follow_states = new List<int>();
    private int current_node_performance_showing = -1;
    private int current_from_to_states_showing = -1;
    private int last_index_on_performance_node_have_state = -1;
    void PlayAnimation()
    {
        if (trajectoryNum != 0 || !flag_play_video)
        {
            return;
        }
        
        if (path_index_follow_states.Count == 0)
        {
            for (int i = HighLevelGuide.testPerformanceNodes.Count - 1; i >= 1 && last_index_on_performance_node_have_state == -1; i--)
            {
                if (HighLevelGuide.testPerformanceNodes[i].initStates.Count > 0)
                {
                    last_index_on_performance_node_have_state = i;
                }
            }

            int cindex_to_follow = 0;
            path_index_follow_states.Add(cindex_to_follow);
            for (int i = last_index_on_performance_node_have_state; i >= 2; i--)
            {
                if (HighLevelGuide.testPerformanceNodes[i].fromIndexInitState.Count > 0)
                {
                    path_index_follow_states.Add(HighLevelGuide.testPerformanceNodes[i].fromIndexInitState[cindex_to_follow]);
                    cindex_to_follow = HighLevelGuide.testPerformanceNodes[i].fromIndexInitState[cindex_to_follow];
                }
            }

            current_node_performance_showing = 1;
            current_from_to_states_showing = 0;
        }

        int index_on_path = path_index_follow_states.Count - 1 - (current_node_performance_showing - 1);

        if (index_on_path < 0)
        {
            current_node_performance_showing = 1;
            index_on_path = path_index_follow_states.Count - 1;
        }

        int index_of_fromToStates = path_index_follow_states[index_on_path];

        if (current_node_performance_showing <= last_index_on_performance_node_have_state)
        {
            if (current_from_to_states_showing < HighLevelGuide.testPerformanceNodes[current_node_performance_showing].fromToStates[index_of_fromToStates].Count)
            {
                int state_id_to_load = HighLevelGuide.testPerformanceNodes[current_node_performance_showing].fromToStates[index_of_fromToStates][current_from_to_states_showing];
                LoadState(state_id_to_load);
                current_from_to_states_showing++;
            }
            else
            {
                current_from_to_states_showing = 0;
                current_node_performance_showing++;
            }
        }
        else
        {
            current_node_performance_showing = 1;
            current_from_to_states_showing = 0;
        }
    }

    private void Update()
    {
        flag_random_shift = !useAgentInterface && !HighLevelGuide.flag_test_performance 
            && HumanoidClimberAcademy.current_training_type != HumanoidClimberAcademy.TrainingType.RandomTraining;

        if (HighLevelGuide.flag_test_performance && HighLevelGuide.flag_show_video_when_path_found
            && (HighLevelGuide.testPerformanceNodes[HighLevelGuide.testPerformanceNodes.Count-1].initStates.Count > 0
            || HighLevelGuide.IsSearchDone))
        {
            flag_play_video = true;
            Time.timeScale = 1;
        }

        if (HighLevelGuide.flag_show_video_when_path_found)
        {
            if (!flag_play_video || (flag_play_video && trajectoryNum != 0))
                SaveInTrajectory();
        }

        PlayAnimation();
        
        if (!useAgentInterface)
            return;

        if (trajectoryNum != 0)
            return;

        Camera.main.GetComponent<CameraFollow>().flagFollowAgent = !useAgentInterface;

        Vector3 nCameraPos = lookAtPos + camera_pos[0] * (new Vector3(Mathf.Cos(camera_pos[1]), Mathf.Sin(camera_pos[2]), Mathf.Sin(camera_pos[1]))).normalized;
        nCameraPos.y = Mathf.Max(0f, nCameraPos.y);

        Camera.main.transform.position = nCameraPos;
        Camera.main.transform.LookAt(lookAtPos, Vector3.up);

        Transform hips = jdController.getBoneTransform(HumanBodyBones.Hips);
        lookAtPos = 0.95f * lookAtPos + 0.05f * hips.position;

        if (Input.GetMouseButton(1))
        {
            Vector3 delta = Input.mousePosition - lastMouseClicked;

            camera_pos[1] += delta[0] / 100.0f;
            camera_pos[2] += delta[1] / 100.0f;

            if (camera_pos[2] < -Mathf.PI / 2.0f)
                camera_pos[2] = -Mathf.PI / 2.0f;
            if (camera_pos[2] > Mathf.PI / 2.0f)
                camera_pos[2] = Mathf.PI / 2.0f;

            if (camera_pos[1] < -Mathf.PI)
                camera_pos[1] = -Mathf.PI;
            if (camera_pos[1] > 0)
                camera_pos[1] = 0;
        }
        lastMouseClicked = Input.mousePosition;

        float mouse_scroll_value = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(mouse_scroll_value) > 0)
        {
            camera_pos[0] += -3.0f * mouse_scroll_value;
        }

        RaycastHit hitInfo = new RaycastHit();
        bool hit = Physics.Raycast(Camera.main.ScreenPointToRay(Input.mousePosition), out hitInfo);
        if (hit)
        {
            if (hitInfo.transform.parent != null)
                if (hitInfo.transform.parent.parent != null && hitInfo.transform.parent.parent.name != "ClimberPair")
                    return;

            if (current_select_state == 0)
            {
                if (hitInfo.transform.gameObject.name == "RHConnector" || hitInfo.transform.gameObject.name == "hand_R")
                {
                    selected_limb = 3;
                }
                else if (hitInfo.transform.gameObject.name == "LHConnector" || hitInfo.transform.gameObject.name == "hand_L")
                {
                    selected_limb = 2;
                }
                else if (hitInfo.transform.gameObject.name == "RLConnector" || hitInfo.transform.gameObject.name == "footR")
                {
                    selected_limb = 1;
                }
                else if (hitInfo.transform.gameObject.name == "LLConnector" || hitInfo.transform.gameObject.name == "footL")
                {
                    selected_limb = 0;
                }
                else
                {
                    selected_limb = -1;
                }
            }
            else
            {
                if (hitInfo.transform.gameObject.tag == "Hold")
                {
                    HoldInfo cHoldInfo = hitInfo.transform.gameObject.GetComponent<HoldInfo>();
                    selected_hold = cHoldInfo.holdId;
                }
                else
                {
                    selected_hold = -1;
                }
            }
        }

        if (Input.GetMouseButton(0))
        {
            if (selected_limb > -1)
            {
                current_select_state = 1;
            }
        }

        if (Input.GetMouseButtonUp(0))
        {
            current_select_state = 0;
            if (selected_limb > -1)
            {
                user_trajectory_points[user_trajectory_points.Count - 1].target_ids[selected_limb] = selected_hold;
                selected_hold = -1;
            }
        }

        for (int h = 0; h < holdPoses.Length; h++)
        {
            SetColorToHolds(h, Color.yellow);
        }

        for (int b = 0; b < 4; b++)
        {
            if (user_trajectory_points[user_trajectory_points.Count - 1].target_ids[b] == connectBodyParts[b].cHoldId)
            {
                SetColorToLimb(b, Color.black);
            }
            else
            {
                SetColorToLimb(b, Color.green);
                if (user_trajectory_points[user_trajectory_points.Count - 1].target_ids[b] > -1)
                {
                    SetColorToHolds(user_trajectory_points[user_trajectory_points.Count - 1].target_ids[b], Color.green);
                    Debug.DrawLine(connectBodyParts[b]._bodyPartInfo.transform.position,
                        holdPoses[user_trajectory_points[user_trajectory_points.Count - 1].target_ids[b]].position);
                }
            }
        }

        if (selected_limb > -1)
        {
            SetColorToLimb(selected_limb, Color.red);
        }

        if (selected_hold > -1)
        {
            SetColorToHolds(selected_hold, Color.red);
        }

        if (Input.GetKeyDown(KeyCode.Return))
        {
            current_command_type = UserCommandType.UserForwardSimulate;

            fatherNodeID = -1;
            graphActionIndex = -1;
            isTaskCompleted = false;
            task_counter_step = counter_agent_running;
            for (int b = 0; b < 4; b++)
            {
                init_hold_ids[b] = connectBodyParts[b].cHoldId;
                connectBodyParts[b].target_hold_id = user_trajectory_points[user_trajectory_points.Count - 1].target_ids[b];
                target_hold_ids[b] = user_trajectory_points[user_trajectory_points.Count - 1].target_ids[b];
            }
            num_spline_done = 0;

            current_count_limb_movment = HumanoidClimberAcademy.Tools.GetDiffBtwSetASetB(target_hold_ids, init_hold_ids) - 1;

            if (agentParameters.onDemandDecision)
            {
                RequestDecision();
            }

            if (useSpline)
            {
                currentDecisionStep = 0;
            }
            else
            {
                currentDecisionStep = 1;
            }
            isNewDecisionStep = true;
            spline_steps_needed = 0;
        }

        if (Input.GetKeyDown(KeyCode.Backspace))
        {
            RemoveLastTrajectoryPoint();
        }
    }

    ///////////////////////////////////End: agent interface////////////////////////////////////

    public int GetCurrentHoldID(int i)
    {
        return connectBodyParts[i].cHoldId;
    }

    public bool IsTaskOnTheLadderClimbing()
    {
        return isTaskOnLadderClimbing;
    }

    public int GetCurrentLimbPlannedIndex()
    {
        return current_idx_limb_planned;
    }

    // set random task from an incomplete graph
    void GetRandomMovementAroundCurState(int fatherNode, bool isTrajectoryForked)
    {
        Vector2Int rnd_index_child_action = HighLevelGuide.graphNodes[fatherNode].GetRandomActionOrChildIndex(current_count_limb_movment, isTrajectoryForked);

        if (rnd_index_child_action[0] > -1)
        {
            // get next target from child index
            int[] cHoldIds = HighLevelGuide.graphNodes[rnd_index_child_action[0]].holdIds;

            for (int h = 0; h < 4; h++)
            {
                connectBodyParts[h].target_hold_id = cHoldIds[h];
            }

            graphActionIndex = -1;
        }
        else if (rnd_index_child_action[1] > -1)
        {
            // get next target from possible actions
            int[] nHoldIDs = HighLevelGuide.graphNodes[fatherNode].GetTargetHoldIDs(current_count_limb_movment, rnd_index_child_action[1]);

            for (int h = 0; h < 4; h++)
            {
                connectBodyParts[h].target_hold_id = nHoldIDs[h];
            }

            graphActionIndex = rnd_index_child_action[1];
        }

        return;
    }

    // set random ladder climbing task from LadderClimbingTaskType
    void SetRandomLadderClimbingTask()
    {
        if (HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.GraphTreeTraining)
        {
            currentTastType = LadderClimbingTaskType.GoUp;
            return;
        }

        if (HumanoidClimberAcademy.current_training_type != HumanoidClimberAcademy.TrainingType.LadderClimbing)
        {
            return;
        }

        not_moved_limbs.Clear();

        float col = 0.0f;
        float row = 0.0f;
        for (int i = 0; i < 4; i++)
        {
            not_moved_limbs.Add(i);
            if (connectBodyParts[i].cHoldId > -1)
            {
                col += (int)(connectBodyParts[i].cHoldId % 4) / 4.0f;
                row += (int)(connectBodyParts[i].cHoldId / 4) / 4.0f;
            }
        }

        List<LadderClimbingTaskType> possible_tasks = new List<LadderClimbingTaskType>();
        possible_tasks.Add(LadderClimbingTaskType.GoDown);
        possible_tasks.Add(LadderClimbingTaskType.GoUp);
        possible_tasks.Add(LadderClimbingTaskType.GoRight);
        possible_tasks.Add(LadderClimbingTaskType.GoLeft);
        if (col < 1.0f)
        {
            possible_tasks.Remove(LadderClimbingTaskType.GoLeft);
        }
        if (col > 2.0f)
        {
            possible_tasks.Remove(LadderClimbingTaskType.GoRight);
        }
        if (row < 1.0f)
        {
            possible_tasks.Remove(LadderClimbingTaskType.GoDown);
        }
        if (row > 2.0f)
        {
            possible_tasks.Remove(LadderClimbingTaskType.GoUp);
        }

        if (col < 1.0f && row < 1.0f)
        {
            HighLevelGuide.AddNode(-1, init_hold_ids, trajectoryNum);
        }
        else if (col < 1.0f && row > 2.0f)
        {
            HighLevelGuide.AddNode(-1, init_hold_ids, trajectoryNum);
        }
        else if (col > 2.0f && row < 1.0f)
        {
            HighLevelGuide.AddNode(-1, init_hold_ids, trajectoryNum);
        }
        else if (col > 2.0f && row > 2.0f)
        {
            HighLevelGuide.AddNode(-1, init_hold_ids, trajectoryNum);
        }

        int rnd_task_id = Random.Range(0, possible_tasks.Count);
        if (rnd_task_id >= 0 && rnd_task_id < possible_tasks.Count)
        {
            currentTastType = possible_tasks[rnd_task_id];
        }
    }

    // set random movement for current ladder climbing task type, each task type is 4 climbing movement
    void PlanForLadderClimbing()
    {
        if (!(HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.GraphTreeTraining
            || HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.LadderClimbing))
        {
            return;
        }
        
        if (current_idx_limb_planned >= 4 || (not_moved_limbs.Count == 0 && HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.LadderClimbing))
        {
            SetRandomLadderClimbingTask();
            current_idx_limb_planned = 0;
        }
        int n_limb_id = -1;
        if (HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.LadderClimbing)
        {
            if (connectBodyParts[2].cHoldId == -1 && connectBodyParts[3].cHoldId == -1)
            {
                if (not_moved_limbs.Count < 3)
                {
                    MyDoneFunc(false, true);
                    not_moved_limbs.Clear();
                    return;
                }
                else
                {
                    n_limb_id = not_moved_limbs[Random.Range(2, not_moved_limbs.Count)];
                    not_moved_limbs.Remove(n_limb_id);
                }
            }
            else
            {
                n_limb_id = not_moved_limbs[Random.Range(0, not_moved_limbs.Count)];
                not_moved_limbs.Remove(n_limb_id);
            }
        }
        else
        {
            int[] limb_order_up = { 2, 3, 0, 1 };
            if (current_idx_limb_planned < 0) current_idx_limb_planned = 0;
            if (current_idx_limb_planned > 3) current_idx_limb_planned = 3;

            n_limb_id = limb_order_up[current_idx_limb_planned];
        }
        current_idx_limb_planned++;

        // plan for next move
        if (connectBodyParts[n_limb_id].target_hold_id == -1)
        {
            // should only happens if the climber is at T-Pos
            int bias = n_limb_id <= 1 ? 0 : 1;
            int limb_to_hold = n_limb_id <= 1 ? n_limb_id : (int)((n_limb_id + 1) / 2) - 1;
            connectBodyParts[n_limb_id].target_hold_id = limb_to_hold + 1 + 4 * bias;
        }
        else
        {
            switch (currentTastType)
            {
                case LadderClimbingTaskType.GoUp:
                    connectBodyParts[n_limb_id].target_hold_id += 4;
                    break;
                case LadderClimbingTaskType.GoDown:
                    connectBodyParts[n_limb_id].target_hold_id -= 4;
                    break;
                case LadderClimbingTaskType.GoRight:
                    connectBodyParts[n_limb_id].target_hold_id += 1;
                    break;
                case LadderClimbingTaskType.GoLeft:
                    connectBodyParts[n_limb_id].target_hold_id -= 1;
                    break;
                default:
                    break;
            }
        }

        return;
    }

    int[] planned_target_ids = { -1, -1, -1, -1 };
    void SetNextTargetHoldIds(bool flag_plan_next_move)
    {
        if (flag_plan_next_move)
        {
            int[] cHoldIDs = new int[] { connectBodyParts[0].cHoldId,
                                         connectBodyParts[1].cHoldId,
                                         connectBodyParts[2].cHoldId,
                                         connectBodyParts[3].cHoldId};

            HighLevelGuide.AddNode(HighLevelGuide.ReturnNodeId(init_hold_ids), cHoldIDs, trajectoryNum, true);

            // initialization for new task
            graphActionIndex = -1;
            isTaskCompleted = false;
            task_counter_step = counter_agent_running;
            for (int b = 0; b < 4; b++)
            {
                init_hold_ids[b] = cHoldIDs[b];
                connectBodyParts[b].target_hold_id = cHoldIDs[b];
                target_hold_ids[b] = cHoldIDs[b];
            }
            num_spline_done = 0;

            current_count_limb_movment = 0;
            if (HumanoidClimberAcademy.max_count_limb_movement == 2)
            {
                float threshold = 0.5f * (HighLevelGuide.loggedStep / 5e6f);
                if (useSpline)
                {
                    threshold = 0.5f * (HighLevelGuide.loggedStep / 10e6f);
                }
                if (threshold > 0.5f)
                    threshold = 0.5f;
                if (Random.Range(0.0f, 1.0f) < threshold)
                {
                    current_count_limb_movment = 1;
                }
            }
            
            // update from-to stance nodes on the stance graph
            fatherNodeID = HighLevelGuide.ReturnNodeId(init_hold_ids);

            HighLevelGuide.UpdateHumanoidState(fatherNodeID, trajectoryNum);
            HighLevelGuide.CompleteNode(fatherNodeID, current_count_limb_movment);

            if (HumanoidClimberAcademy.Tools.GetNumConnectedLimbs(planned_target_ids) == 0)
            {
                if (isTrajectoryForked && (HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.GraphTreeTraining
                    || HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.LadderClimbing))
                {
                    float randomly_do_goup = 0.0f;
                    if (HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.GraphTreeTraining)
                    {
                        randomly_do_goup = Random.Range(0.0f, 1.0f);
                        if (randomly_do_goup < 0.85f || !isTaskOnLadderClimbing)
                        {
                            if (fatherNodeID == -1)
                            {
                                MyDoneFunc(false, true);
                                return;
                            }

                            GetRandomMovementAroundCurState(fatherNodeID, isTrajectoryForked);

                            isTaskOnLadderClimbing = false;
                        }
                    }

                    if ((HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.LadderClimbing || randomly_do_goup >= 0.85f) && isTaskOnLadderClimbing)
                    {
                        PlanForLadderClimbing();
                    }
                }
                else
                {
                    isTaskOnLadderClimbing = false;

                    GetRandomMovementAroundCurState(fatherNodeID, isTrajectoryForked);
                }
            }
            else
            {
                for (int b = 0; b < 4; b++)
                {
                    connectBodyParts[b].target_hold_id = planned_target_ids[b];
                }
            }

            for (int b = 0; b < 4; b++)
            {
                if (connectBodyParts[b].target_hold_id != connectBodyParts[b].cHoldId && connectBodyParts[b].target_hold_id >= 0)
                    task_planned_num++;
                target_hold_ids[b] = connectBodyParts[b].target_hold_id;
            }
            current_count_limb_movment = HumanoidClimberAcademy.Tools.GetDiffBtwSetASetB(target_hold_ids, init_hold_ids) - 1;

            if (!HighLevelGuide.flag_test_performance)
            {
                if (!HumanoidClimberAcademy.IsValidTransition(init_hold_ids, target_hold_ids))
                {
                    for (int b = 0; b < 4; b++)
                    {
                        connectBodyParts[b].target_hold_id = connectBodyParts[b].cHoldId;
                    }
                    return;
                }
            }
            // randomize target hold position around the agent
            if (HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.UniformAroundStateTraining)
            {
                Vector3 hipLocation = jdController.bones[0].rb.transform.localPosition;
                
                HighLevelGuide.GetRandomPositionAroundAgent(hipLocation, cHoldIDs, target_hold_ids, holdPoses);
            }
            else
            {
                // check distance validity 
                Vector3 hipLocation = jdController.bones[0].rb.transform.localPosition;
                if (!HighLevelGuide.HasValidDistance(hipLocation, target_hold_ids, holdPoses))
                {
                    for (int b = 0; b < 4; b++)
                    {
                        connectBodyParts[b].target_hold_id = connectBodyParts[b].cHoldId;
                    }
                    return;
                }

            }
            
            // during training, we do not want to have repeated movements (we want more diverse movements)
            if (isTrajectoryForked)
            {
                ulong stanceNodeID = HumanoidClimberAcademy.StanceToKey(target_hold_ids);
                if (agent_tree_task.Contains(stanceNodeID))
                {
                    MyDoneFunc(false, true);
                }
                else
                {
                    agent_tree_task.Add(stanceNodeID);
                }
            }

            for (int b = 0; b < 4; b++)
            {
                if (connectBodyParts[b].target_hold_id >= 0)
                {
                    current_dis_target[b] = (holdPoses[connectBodyParts[b].target_hold_id].localPosition
                                - connectBodyParts[b]._bodyPartInfo.transform.localPosition).magnitude - 0.25f - 0.15f;
                }
                else
                {
                    current_dis_target[b] = float.MaxValue;
                }
                init_dis_target[b] = current_dis_target[b];
            }

        }

        return;
    }

    void ApplyControl(float[] vectorAction)
    {
        //float[] cAngles = new float[jdController.numControlDOFs()];
        //float[] cAngleRates = new float[jdController.numControlDOFs()];
        //float c = HighLevelGuide.KAngleRate;
        //float maxRate = c * Mathf.PI * Mathf.Rad2Deg * Time.fixedDeltaTime * 5f;
        //if (useAngleSmoothing)
        //{
        //    jdController.poseToMotorAngles(ref cAngles);
        //    jdController.getMotorAngleRates(ref cAngleRates);
        //}
        
        int i = -1;
        for (int j = 0; j < jdController.numControlDOFs(); j++)
        {
            i++;
            if (!useSpline)
            {
                float f = (vectorAction[i] + 1f) * 0.5f * (jdController.maxAngle[j] - jdController.minAngle[j]) + jdController.minAngle[j];
                //const float predictionTime = 0.1f;// 5 * Time.fixedDeltaTime
                //float angle_prediction = cAngles[j] + cAngleRates[j] * predictionTime;
                //angle_prediction = Mathf.Clamp(angle_prediction, jdController.minAngle[j], jdController.maxAngle[j]);
                //if (useAngleSmoothing)
                //{
                //    //float diff = (f - angle_prediction);
                //    //if (Mathf.Abs(diff) > maxRate)
                //    //    diff = Mathf.Sign(diff) * maxRate;
                //    sampleVectorAction[i] = (1 - c) * angle_prediction + c * f;
                //    sampleVectorAction[i] = Mathf.Clamp(sampleVectorAction[i], jdController.minAngle[j], jdController.maxAngle[j]);
                //}
                //else
                //{
                sampleVectorAction[i] = f;
                //}
            }
            else
            {
                sampleVectorAction[i] = vectorAction[i];
            }
        }
        if (trajectoryNum == 0)
        {
            int notifyme = 1;
        }
        jdController.driveToPose(sampleVectorAction, 0);

        for (int b = 0; b < jdController.getNumBones(); b++)
        {
            if (jdController.bones[b].amotor != null || jdController.bones[b].hinge != null)
            {
                i++;
                if (!useSpline)
                {
                    float f = (vectorAction[i] + 1f) * 0.5f * HumanoidMecanimRig.maxJointForceLimit;
                    sampleVectorAction[i] = f;
                    jdController.bones[b].SetJointStrength(f);
                }
                else
                {
                    sampleVectorAction[i] = vectorAction[i];
                    jdController.bones[b].SetJointStrength(sampleVectorAction[i]);
                }
            }
        }

        return;
    }

    void GenerateControlValues(int step)
    {
        // this function re-create spline given 'step' and (v,dv) at 'step' of spline
        // assuming we are at correct step of spline (i.e. spline v,dv are set properly)
        //, then given 'step' we can re-create same spline

        // maxStep calculated such that step is not violating segmentID

        if (trajectoryNum == 0)
        {
            int notifyme = 1;
        }

        float[] sample = _sampledActionSpline;
        int nValuesPerSegment = 1 + nActionSize;

        // total passed time
        float totalTime = step * Time.fixedDeltaTime;

        // restarting segmentIDs in step = 0
        int segmentIdx = 0;
        if (step == 0)
            segmentIdx = 0;
        else
        {
            float s_t1 = sample[0 * nValuesPerSegment];
            float s_t2 = sample[Mathf.Min(1, nControlPoints - 1) * nValuesPerSegment];
            if (s_t1 < totalTime)
                segmentIdx = 1;
            else if (s_t1 + s_t2 < totalTime)
                segmentIdx = 2;
        }

        // calculating t1 based on all segments and total passed time
        float t1 = 0.0f;
        for (int _sID = 0; _sID <= segmentIdx; _sID++)
            t1 += sample[_sID * nValuesPerSegment];
        t1 = t1 - totalTime;

        float t2 = t1 + sample[Mathf.Min(segmentIdx + 1, nControlPoints - 1) * nValuesPerSegment];
        if (t1 < 0)
        {
            t1 = t2 - Time.fixedDeltaTime;
            segmentIdx++;

            t2 = t1 + sample[Mathf.Min(segmentIdx + 1, nControlPoints - 1) * nValuesPerSegment];
        }

        float maxSpeed = 2.0f * Mathf.Rad2Deg * Mathf.PI;
        for (int i = 0; i < nActionSize; i++)
        {
            float s1 = sample[Mathf.Min(segmentIdx, nControlPoints - 1) * nValuesPerSegment + 1 + i],
                  s2 = sample[Mathf.Min(segmentIdx + 1, nControlPoints - 1) * nValuesPerSegment + 1 + i];

            float p1 = ((s1 + 1.0f) / 2.0f) * (MinMaxControlValues[i][1] - MinMaxControlValues[i][0]) + MinMaxControlValues[i][0];
            float p2 = ((s2 + 1.0f) / 2.0f) * (MinMaxControlValues[i][1] - MinMaxControlValues[i][0]) + MinMaxControlValues[i][0];

            float cVal = spline[i].currentValue;
            spline[i].step(Time.fixedDeltaTime, p1, t1, p2, t2);

            if (i < jdController.numControlDOFs())
            {
                spline[i].setState(Mathf.Clamp(spline[i].currentValue, MinMaxControlValues[i][0], MinMaxControlValues[i][1]),
                                   Mathf.Clamp(spline[i].currentDerivativeValue, -maxSpeed, maxSpeed));
                if (Mathf.Abs(spline[i].currentValue - cVal) > maxSpeed)
                {
                    spline[i].setCurrentValue(spline[i].currentValue + Mathf.Sign(spline[i].currentValue - cVal) * maxSpeed);
                }
            }
            else
            {
                spline[i].setState(Mathf.Clamp(spline[i].currentValue, MinMaxControlValues[i][0], MinMaxControlValues[i][1]),
                                   Mathf.Clamp(spline[i].currentDerivativeValue, -1000.0f, 1000.0f));
            }
            interpolatedControl[i] = spline[i].currentValue;

            if (float.IsNaN(interpolatedControl[i]))
            {
                interpolatedControl[i] = 0.0f;
                spline[i].setState(0.0f, 0.0f);
            }
        }

        return;
    }

    /// <summary>
    /// when this function is called the agent will end the current episode and restart
    /// </summary>
    /// <param name="is_failure"></param>
    bool reset_agent_state_needed = false;
    public override void MyDoneFunc(bool is_failure, bool _reset_needed)
    {
        if (!useAgentInterface)
        {
            if (!IsDone())
            {
                if (is_failure)
                {
                    HighLevelGuide.AddFailure(fatherNodeID, graphActionIndex, current_count_limb_movment);
                }
                Done();
            }

            if (!reset_agent_state_needed)
                reset_agent_state_needed = is_failure || _reset_needed;

            base.MyDoneFunc(is_failure, _reset_needed);
        }
        return;
    }

    bool IsAgentReachedTargetHolds()
    {
        for (int b = 0; b < 4; b++)
        {
            if (connectBodyParts[b].target_hold_id == connectBodyParts[b].cHoldId && connectBodyParts[b].target_hold_id >= 0
                && init_hold_ids[b] != connectBodyParts[b].cHoldId)
            {
                return true;
            }
        }

        return false;
    }

    float GetDisStrength(float[] vectorAction)
    {
        float dis = 0;
        int strenght_length = vectorAction.Length - jdController.numControlDOFs();
        for (int i = 0; i < strenght_length; i++)
        {
            float strength = vectorAction[vectorAction.Length - 1 - i];
            var rawVal = (strength + 1f) * 0.5f;

            dis += rawVal * rawVal;
        }
        return dis / (float)strenght_length; //return the average squared normalized motor strength
    }

    void SetMinMaxXontrolValues()
    {
        for (int j = 0; j < jdController.numControlDOFs(); j++)
        {
            MinMaxControlValues.Add(new Vector2(jdController.minAngle[j], jdController.maxAngle[j]));
        }

        //update joint strength settings
        for (int j = 0; j < jdController.getNumBones(); j++)
        {
            MinMaxControlValues.Add(new Vector2(0, HumanoidMecanimRig.maxJointForceLimit));
        }

        return;
    }
    ////////////////////////////////////////////////// Climber Agent Override Functions /////////////////////////////////////////

    void QuitApp()
    {
#if UNITY_EDITOR
        // Application.Quit() does not work in the editor so
        // UnityEditor.EditorApplication.isPlaying need to be set to false to end the game
        UnityEditor.EditorApplication.isPlaying = false;
#endif
        Application.Quit();

    }

    public override void InitializeAgent()
    {
        if (!HighLevelGuide.isInitialized)
            HighLevelGuide.InitializeAcademy();
        jdController = GetComponent<HumanoidMecanimRig>();

        if (!jdController.initialized)
            jdController.initialize();
        
        lookAtPos = jdController.getWorldBonePos(HumanBodyBones.Hips);// hips.position;

        if (HighLevelGuide.flag_test_performance)
        {
            StreamReader mRouteNumFile = new StreamReader("ClimberRouteInfo\\RouteNum.txt");
            int route_num = int.Parse(mRouteNumFile.ReadLine());
            mRouteNumFile.Close();

            StreamReader mRouteHoldPositionFile = new StreamReader("ClimberRouteInfo\\mHoldsRoute" + route_num.ToString() + ".txt");
            int h = 0;
            Vector3 nV = new Vector3();
            while (!mRouteHoldPositionFile.EndOfStream)
            {
                nV = HumanoidClimberAcademy.Tools.StringToVector3(mRouteHoldPositionFile.ReadLine());
                if (h < holdPoses.Length)
                {
                    Vector3 nPos = new Vector3(localAgentScaling * nV.x - 4.0f, localAgentScaling * nV.z - 5.6f, 3.875f);
                    holdPoses[h].transform.localPosition = HighLevelGuide.GetRestrictedHoldPos(nPos);
                    h++;
                }
            }
            for (; h < holdPoses.Length; h++)
            {
                holdPoses[h].transform.localPosition = new Vector3(10f, 7.12f, 3.875f);
            }
            mRouteHoldPositionFile.Close();

            // replacing the agent
            StreamReader mAgentPositionFile = new StreamReader("ClimberRouteInfo\\mClimberInfoFile" + route_num.ToString() + ".txt");
            Vector3 relocate_dis = new Vector3(localAgentScaling * float.Parse(mAgentPositionFile.ReadLine()) - 4.0f, -6.0f, -1.5f) - base.transform.localPosition;
            base.transform.localPosition = base.transform.localPosition + relocate_dis;
            LLConnector.localPosition += relocate_dis;
            RLConnector.localPosition += relocate_dis;
            LHConnector.localPosition += relocate_dis;
            RHConnector.localPosition += relocate_dis;
            mAgentPositionFile.Close();
        }

        if (useAgentInterface)
        {
            agentParameters.maxStep = 0;
        }

        current_step = -1;

        while (HighLevelGuide._ControlRigs.Count < 16)
        {
            HighLevelGuide._ControlRigs.Add(null);
        }

        if (connectBodyParts.Count == 0)
        {
            for (int i = 0; i < 4; i++)
            {
                switch (i)
                {
                    case 0:
                        connectBodyParts.Add(new EndBodyPart(LLConnector, this));
                        break;
                    case 1:
                        connectBodyParts.Add(new EndBodyPart(RLConnector, this));
                        break;
                    case 2:
                        connectBodyParts.Add(new EndBodyPart(LHConnector, this));
                        break;
                    case 3:
                        connectBodyParts.Add(new EndBodyPart(RHConnector, this));
                        break;
                }
            }
        }

        if (useSpline)
        {
            if (spline.Count == 0)
            {
                for (int i = 0; i < nActionSize; i++)
                {
                    spline.Add(new RecursiveTCBSpline());
                    spline[i].setState(0.0f, 0.0f);
                    if (useLinearSpline)
                        spline[i].linearMix = 1.0f;
                    else
                        spline[i].linearMix = 0.0f;
                }
            }
            interpolatedControl = new float[nActionSize];

            _sampledActionSpline = new float[nControlPoints * (1 + nActionSize)];
            splineStoredVectorAction = new float[nControlPoints * (1 + nActionSize)];

            //agentParameters.onDemandDecision = true;
            agentParameters.numberOfActionsBetweenDecisions = 10;
            SetMinMaxXontrolValues();
        }
        else if (HumanoidClimberAcademy.MultiStepAction)
        {
            agentParameters.numberOfActionsBetweenDecisions = 20;
        }
        else
        {
            agentParameters.numberOfActionsBetweenDecisions = 5;
        }

        sampleVectorAction = new float[nActionSize];
        
        if (agentParameters.onDemandDecision)
            RequestDecision();

        fatherNodeID = -1;
        graphActionIndex = -1;
        for (int b = 0; b < 4; b++)
        {
            init_hold_ids[b] = connectBodyParts[b].cHoldId;
            target_hold_ids[b] = connectBodyParts[b].cHoldId;
        }

        if (trajectoryNum == -1)
        {
            m_agent_counter++;
            trajectoryNum = m_agent_counter;
        }

        isTaskOnLadderClimbing = true;
        current_idx_limb_planned = 0;
        HighLevelGuide._ControlRigs[trajectoryNum] = this;

        counter_agent_running = 0;

        if (!useAgentInterface)
        {
            HighLevelGuide.AddNode(-1, init_hold_ids, trajectoryNum, false);
            SaveInTrajectory();
        }
        else
        {
            AddTrajectoryPoint();
        }

        if (HighLevelGuide.flag_test_performance)
        {
            fromInitIndexPerformanceNodeIds[0] = -1;
            fromInitIndexPerformanceNodeIds[1] = -1;
            HighLevelGuide.CompleteTestPerformanceNode(agent_state_ids, current_step, trajectoryNum, fromInitIndexPerformanceNodeIds, true);
        }
        return;
    }

    private void MyAddVectorObs(Vector3 observation)
    {
        if (float.IsNaN(observation.magnitude) || float.IsInfinity(observation.magnitude))
        {
            AddVectorObs(Vector3.zero);
            MyDoneFunc(true, true);
        }
        else
        {
            AddVectorObs(observation);
        }
    }

    private void MyAddVectorObs(float observation)
    {
        if (float.IsNaN(observation) || float.IsInfinity(observation))
        {
            AddVectorObs(0f);
            MyDoneFunc(true, true);
        }
        else
        {
            AddVectorObs(observation);
        }
    }

    public override void CollectObservations()
    {
        jdController.GetCurrentJointForces();

        Transform hips = jdController.getBoneTransform(HumanBodyBones.Hips);
        Transform handL = jdController.getBoneTransform(HumanBodyBones.LeftHand);
        Transform handR = jdController.getBoneTransform(HumanBodyBones.RightHand);
        Transform footL = jdController.getBoneTransform(HumanBodyBones.LeftFoot);
        Transform footR = jdController.getBoneTransform(HumanBodyBones.RightFoot);
        Transform head = jdController.getBoneTransform(HumanBodyBones.Head);
        MyAddVectorObs(hips.forward);
        MyAddVectorObs(hips.up);

        foreach (var bodyPart in jdController.bones)
        {
            var rb = bodyPart.rb;
            if (bodyPart.rb.transform != handL && bodyPart.rb.transform != handR)
            {
                MyAddVectorObs((bodyPart.groundContact.touchingGround ? 1 : 0)); // Is this bp touching the ground
                MyAddVectorObs((bodyPart.wallContact.touchingWall ? 1 : 0)); // Is this bp touching the wall
            }
            MyAddVectorObs(rb.velocity);
            MyAddVectorObs(rb.angularVelocity);
            if (bodyPart.rb.transform != hips)
            {
                Vector3 localPosRelToHips = hips.InverseTransformPoint(rb.position);
                MyAddVectorObs(localPosRelToHips);
            }

            if (bodyPart.rb.transform != hips && bodyPart.rb.transform != handL && bodyPart.rb.transform != handR &&
                bodyPart.rb.transform != footL && bodyPart.rb.transform != footR && bodyPart.rb.transform != head)
            {
                MyAddVectorObs(bodyPart.currentXNormalizedRot);
                MyAddVectorObs(bodyPart.currentYNormalizedRot);
                MyAddVectorObs(bodyPart.currentZNormalizedRot);
                MyAddVectorObs(bodyPart.currentStrength / (HumanoidMecanimRig.maxJointForceLimit + 1e-6f));
            }
        }

        // which body parts are moving
        for (int b = 0; b < 4; b++)
        {
            MyAddVectorObs(init_hold_ids[b] != target_hold_ids[b] ? 1f : 0f);
        }

        for (int b = 0; b < 4; b++)
        {
            MyAddVectorObs(connectBodyParts[b].cHoldId >= 0 ? 1f : 0f);
            MyAddVectorObs(connectBodyParts[b].target_hold_id >= 0 ? 1f : 0f);

            MyAddVectorObs(connectBodyParts[b].groundContact.touchingGround ? 1f : 0f); // Is this bp touching the ground
            MyAddVectorObs(connectBodyParts[b].wallContact.touchingWall ? 1f : 0f); // Is this bp touching the wall
            if (connectBodyParts[b].target_hold_id >= 0 && connectBodyParts[b].target_hold_id < holdPoses.Length)
            {
                Vector3 localPosRelToHips = hips.InverseTransformPoint(holdPoses[connectBodyParts[b].target_hold_id].position);
                MyAddVectorObs(localPosRelToHips);
            }
            else
            {
                Vector3 localPosRelToHips = new Vector3(-1f, -1f, -1f);
                MyAddVectorObs(localPosRelToHips);
            }
        }
        //232 up until here
        if (useObstacles)
        {
            for (int o = 0; o < Obstacles.Length; o++)
            {
                Vector3 localPosRelToHips = hips.InverseTransformPoint(Obstacles[o].position);
                MyAddVectorObs(localPosRelToHips);
                MyAddVectorObs(Obstacles[o].rotation * Vector3.right);
                MyAddVectorObs(Obstacles[o].localScale);
            }
        }
        if (useSpline)
        {
            for (int i = 0; i < nActionSize; i++)
            {
                if (float.IsNaN(spline[i].currentValue) || float.IsNaN(spline[i].currentDerivativeValue))
                {
                    spline[i].setState(0f, 0f);
                }
                MyAddVectorObs(spline[i].currentValue);
                MyAddVectorObs(spline[i].currentDerivativeValue);
            }
        }
        return;
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if (flag_play_video && trajectoryNum == 0)
        {
            return;
        }

        for (int i = 0; i < vectorAction.Length; i++)
        {
            if (float.IsNaN(vectorAction[i]))
            {
                vectorAction[i] = 0f;
                QuitApp();
            }
        }

        bool isTargetSet = false;
        
        for (int i = 0; i < 4 && !isTargetSet; i++)
        {
            if (connectBodyParts[i].cHoldId != connectBodyParts[i].target_hold_id)
            {
                isTargetSet = true;
            }
        }

        if (useSpline)
        {
            if (currentDecisionStep >= spline_steps_needed)
            {
                if (isNewDecisionStep)
                {
                    for (int i = 0; i < vectorAction.Length; i++)
                    {
                        if (float.IsNaN(vectorAction[i]))
                        {
                            vectorAction[i] = 0.0f;
                        }
                        vectorAction[i] = Mathf.Clamp(vectorAction[i], -1f, 1f);
                        splineStoredVectorAction[i] = vectorAction[i];
                        _sampledActionSpline[i] = vectorAction[i];
                    }

                    if (currentDecisionStep > 0)
                    {
                        AddReward(-0.01f);//if we use 50 1-step-splines (50*10 sim steps), we gather -0.1f reward for whole trajectory!
                        num_spline_done++;
                    }
                    currentDecisionStep = 0;
                    counter_spline_not_updated = 0;
                    spline_steps_needed = (minSplineStep + (int)(((vectorAction[0] + 1f) / 2f) * (maxSplineStep - minSplineStep))) * agentParameters.numberOfActionsBetweenDecisions;
                    
                    _sampledActionSpline[0] = spline_steps_needed * Time.fixedDeltaTime;
                    _sampledActionSpline[1 + nActionSize] = 
                        (minSplineStep + (int)(((_sampledActionSpline[1 + nActionSize] + 1f) / 2f) * (maxSplineStep - minSplineStep))) 
                        * agentParameters.numberOfActionsBetweenDecisions * Time.fixedDeltaTime;
                }
            }
            else
            {
                if (isNewDecisionStep)
                {
                    _sampledActionSpline[0] -= Time.fixedDeltaTime * agentParameters.numberOfActionsBetweenDecisions;

                    if (_sampledActionSpline[0] < agentParameters.numberOfActionsBetweenDecisions * Time.fixedDeltaTime)
                        _sampledActionSpline[0] = agentParameters.numberOfActionsBetweenDecisions * Time.fixedDeltaTime;

                    int s0 = spline_steps_needed - (currentDecisionStep - 1);
                    vectorAction[0] = Mathf.Clamp((((s0 / (float)agentParameters.numberOfActionsBetweenDecisions) - minSplineStep) / (float)(maxSplineStep - minSplineStep)) * 2f - 1f, -1f, 1f);
                    //_sampledActionSpline[0] = vectorAction[0];
                    for (int i = 1; i < vectorAction.Length; i++)
                    {
                        vectorAction[i] = splineStoredVectorAction[i];
                        //_sampledActionSpline[i] = splineStoredVectorAction[i];
                    }

                    counter_spline_not_updated = 0;
                }
                else
                {
                    counter_spline_not_updated++;
                }
            }
        }

        if (!useAgentInterface)
        {
            HighLevelGuide.CompleteGraphGradually(current_count_limb_movment);

            if (isTargetSet)
            {
                succeeded_transition = false;
            }
            // if previous task is completed, this one does not count freeing limbs as a movement
            if (IsAgentReachedTargetHolds())
            {
                int[] cHoldIDs = new int[] { connectBodyParts[0].cHoldId,
                                         connectBodyParts[1].cHoldId,
                                         connectBodyParts[2].cHoldId,
                                         connectBodyParts[3].cHoldId};

                HighLevelGuide.AddNode(fatherNodeID, cHoldIDs, trajectoryNum, true);

                // reached to the target, get reward
                // reward only based on the moving the limbs not letting go or freeing (////////TODO: remove this part if the agent is deciding on letting go)
                isTaskCompleted = true;
                for (int b = 0; b < 4; b++)
                {
                    if (connectBodyParts[b].target_hold_id == connectBodyParts[b].cHoldId &&
                        init_hold_ids[b] != connectBodyParts[b].cHoldId && connectBodyParts[b].cHoldId >= 0)
                    {
                        init_hold_ids[b] = connectBodyParts[b].cHoldId;
                        AddReward(1f);
                    }

                    if (connectBodyParts[b].target_hold_id != connectBodyParts[b].cHoldId)
                    {
                        isTaskCompleted = false;
                    }
                }

                if (isTaskCompleted)
                {
                    HighLevelGuide.AddSuccess(fatherNodeID, graphActionIndex, current_count_limb_movment);
                    isTargetSet = false;
                    succeeded_transition = true;
                    if (reset_each_transition)
                        MyDoneFunc(false, false);
                    
                    // during training, we do not want to focuse more on one path
                    if (task_planned_num > HumanoidClimberAcademy.max_task_in_episode)
                        MyDoneFunc(false, true);
                }
                if (!reset_each_transition && !HighLevelGuide.flag_show_video_when_path_found)
                    SaveInTrajectory();
            }
            else
            {
                if (isTargetSet && (counter_agent_running - task_counter_step > 500)) // || num_spline_done >= max_spline_used
                {
                    int[] cHoldIDs = new int[] { connectBodyParts[0].cHoldId,
                                             connectBodyParts[1].cHoldId,
                                             connectBodyParts[2].cHoldId,
                                             connectBodyParts[3].cHoldId};

                    task_failed_num++;

                    if (cHoldIDs[2] > -1 || cHoldIDs[3] > -1)
                    {
                        HighLevelGuide.AddFailure(fatherNodeID, graphActionIndex, current_count_limb_movment);
                        HighLevelGuide.AddNode(fatherNodeID, cHoldIDs, trajectoryNum, true);

                        for (int i = 0; i < 4; i++)
                        {
                            connectBodyParts[i].target_hold_id = connectBodyParts[i].cHoldId;
                        }

                        isTaskCompleted = true;
                        isTargetSet = false;

                        if (isTaskOnLadderClimbing)
                        {
                            current_idx_limb_planned--;
                            if (current_idx_limb_planned < 0)
                            {
                                current_idx_limb_planned = 4;
                            }
                        }

                        if (!reset_each_transition)
                            SaveInTrajectory();
                    }
                    else
                    {
                        MyDoneFunc(true, true);
                    }

                    if (reset_each_transition)
                        MyDoneFunc(false, false);
                    
                    if (task_failed_num > HumanoidClimberAcademy.max_num_fails_in_episode)
                    {
                        MyDoneFunc(false, true);
                    }
                    if (task_planned_num > HumanoidClimberAcademy.max_task_in_episode)
                        MyDoneFunc(false, true);
                }
            }

            if (!reset_each_transition)
            {
                SetNextTargetHoldIds(!isTargetSet);
            }
            else if (!isTargetSet)
            {
                MyDoneFunc(false, false);
            }
            // Apply action to all relevant body parts. 
            if (isNewDecisionStep)
            {
                if (!useSpline)
                {
                    ApplyControl(vectorAction);
                }
                else
                {
                    if (agentParameters.onDemandDecision)
                        agentParameters.numberOfActionsBetweenDecisions = spline_steps_needed;// (int)(_sampledActionSpline[0] / Time.fixedDeltaTime);
                }
            }

            if (useSpline)
            {
                GenerateControlValues(counter_spline_not_updated);//currentDecisionStep
                //if (currentDecisionStep % agentParameters.numberOfActionsBetweenDecisions == 0)
                //{
                ApplyControl(interpolatedControl);
                //}
            }

            ApplyConnectDisconnect();

            // Add relative reward to the current dis to the target
            if (isTargetSet)
            {
                float k_strenght = 0.2f;//0.2f
                float k_target = 1.0f - k_strenght;
                bool flag_add_strenght_reward = false;
                for (int b = 0; b < 4; b++)
                {
                    if (connectBodyParts[b].target_hold_id >= 0 && !connectBodyParts[b].IsConnected)
                    {
                        float c_dis = (holdPoses[connectBodyParts[b].target_hold_id].localPosition
                            - connectBodyParts[b]._bodyPartInfo.transform.localPosition).magnitude - 0.25f - 0.15f;

                        if (trajectoryNum == 0)
                        {
                            Vector3 _dir = connectBodyParts[b]._bodyPartInfo.transform.forward;
                            if (b == 3)
                                _dir = connectBodyParts[b]._bodyPartInfo.transform.right;
                            else if (b == 2)
                                _dir = -connectBodyParts[b]._bodyPartInfo.transform.right;
                            Debug.DrawLine(connectBodyParts[b]._bodyPartInfo.transform.position, holdPoses[connectBodyParts[b].target_hold_id].position);
                            Debug.DrawLine(connectBodyParts[b]._bodyPartInfo.transform.position, connectBodyParts[b]._bodyPartInfo.transform.position + connectBodyParts[b].GetRigidBody().velocity.normalized, Color.red);
                            Debug.DrawLine(connectBodyParts[b]._bodyPartInfo.transform.position, connectBodyParts[b]._bodyPartInfo.transform.position + _dir, Color.blue);
                        }

                        switch (current_reward_type)
                        {
                            case RewardType.DeepMimicDistance:
                                if (useSpline)
                                {
                                    AddReward(Mathf.Exp(-4.0f * c_dis) / (float)(spline_steps_needed + 1e-6f));
                                }
                                else
                                {
                                    float targetDisReward = Mathf.Exp(-4.0f * c_dis) / (float)(agentParameters.numberOfActionsBetweenDecisions + 1e-6);
                                    AddReward(k_target * targetDisReward);
                                }
                                break;
                            case RewardType.DeepMimicThreshold:
                                if (c_dis < current_dis_target[b])
                                {
                                    if (HighLevelGuide.useSameRewarding)
                                    {
                                        float targetDisReward = Mathf.Exp(-4.0f * c_dis) / (float)(5f + 1e-6);
                                        AddReward(k_target * targetDisReward);
                                    }
                                    else
                                    {
                                        if (useSpline)
                                        {
                                            AddReward(Mathf.Exp(-4.0f * c_dis) / (float)(spline_steps_needed + 1e-6f));
                                        }
                                        else
                                        {
                                            float targetDisReward = Mathf.Exp(-4.0f * c_dis) / (float)(agentParameters.numberOfActionsBetweenDecisions + 1e-6);
                                            AddReward(k_target * targetDisReward);
                                        }
                                    }
                                    current_dis_target[b] = c_dis;
                                    flag_add_strenght_reward = true;
                                }
                                break;
                            case RewardType.Percentage:
                                if (c_dis < current_dis_target[b])
                                {
                                    float targetDisReward = (current_dis_target[b] - c_dis) / (init_dis_target[b] + 1e-6f);
                                    AddReward(targetDisReward);
                                    
                                    current_dis_target[b] = c_dis;
                                    flag_add_strenght_reward = true;
                                }
                                break;
                            default:
                                break;
                        }

                        
                    }
                }
                if (flag_add_strenght_reward)
                {
                    float dis_Strength = GetDisStrength(vectorAction);
                    float desiredMaxStrength = 0.05f;
                    if (HighLevelGuide.useSameRewarding)
                    {
                        float strengthReward = Mathf.Exp(-0.5f * dis_Strength / (desiredMaxStrength * desiredMaxStrength)) / (float)(5.0f + 1e-6);
                        AddReward(k_strenght * strengthReward);
                    }
                    else
                    {
                        float strengthReward = Mathf.Exp(-0.5f * dis_Strength / (desiredMaxStrength * desiredMaxStrength)) / (float)(agentParameters.numberOfActionsBetweenDecisions + 1e-6);
                        AddReward(k_strenght * strengthReward);
                    }
                }
            }

            //if (trajectoryNum == 0)
            //{
            //    Debug.Log(GetReward().ToString() + "," + currentDecisionStep.ToString() + "," + _diff.ToString());
            //}

            if (agentParameters.onDemandDecision)
            {
                RequestAction();
            }
            // increasing time
            IncrementDecisionTimer();
        }
        else
        {
            if (current_command_type != UserCommandType.UserForwardSimulate)
            {
                LoadState(user_trajectory_points[user_trajectory_points.Count - 1].cStateID);
                if (agentParameters.onDemandDecision)
                {
                    RequestAction();
                }
                return;
            }

            for (int i = 0; i < 4; i++)
            {
                connectBodyParts[i].target_hold_id = user_trajectory_points[user_trajectory_points.Count - 1].target_ids[i];
                target_hold_ids[i] = connectBodyParts[i].target_hold_id;
            }

            // if previous task is completed, this one does not count freeing limbs as a movement
            if (IsAgentReachedTargetHolds())
            {
                int[] cHoldIDs = new int[] { connectBodyParts[0].cHoldId,
                                         connectBodyParts[1].cHoldId,
                                         connectBodyParts[2].cHoldId,
                                         connectBodyParts[3].cHoldId};

                isTaskCompleted = true;
                for (int b = 0; b < 4; b++)
                {
                    init_hold_ids[b] = connectBodyParts[b].cHoldId;
                    if (connectBodyParts[b].target_hold_id != connectBodyParts[b].cHoldId)
                    {
                        isTaskCompleted = false;
                    }
                }

                if (isTaskCompleted)
                {
                    if (useSpline)
                        Debug.Log("spline:" + num_spline_done.ToString());

                    current_command_type = UserCommandType.UserNone;
                    AddTrajectoryPoint();
                }
            }
            else
            {
                if (counter_agent_running - task_counter_step > 500) //|| num_spline_done > max_spline_used
                {
                    if (useSpline)
                        Debug.Log("spline:" + num_spline_done.ToString());

                    isTaskCompleted = true;
                    current_command_type = UserCommandType.UserNone;
                    AddTrajectoryPoint();

                    int[] cHoldIDs = new int[] { connectBodyParts[0].cHoldId,
                                                 connectBodyParts[1].cHoldId,
                                                 connectBodyParts[2].cHoldId,
                                                 connectBodyParts[3].cHoldId};

                    if (cHoldIDs[2] > -1 || cHoldIDs[3] > -1)
                    {
                        HighLevelGuide.AddFailure(fatherNodeID, graphActionIndex, current_count_limb_movment);
                        HighLevelGuide.AddNode(fatherNodeID, cHoldIDs, trajectoryNum, true);

                        for (int i = 0; i < 4; i++)
                        {
                            connectBodyParts[i].target_hold_id = connectBodyParts[i].cHoldId;
                        }
                    }
                }
            }

            // Apply action to all relevant body parts. 
            if (isNewDecisionStep)
            {
                if (!useSpline)
                {
                    ApplyControl(vectorAction);
                }
                else
                {
                    //int nValuesPerSegment = 1 + nActionSize;
                    //for (int i = 0; i < vectorAction.Length; i++)
                    //{
                    //    if (float.IsNaN(vectorAction[i]))
                    //    {
                    //        vectorAction[i] = 0.0f;
                    //    }
                    //    _sampledActionSpline[i] = Mathf.Clamp(vectorAction[i], -1f, 1f);
                    //}
                    //_sampledActionSpline[0] = (minSplineStep + (int)(((_sampledActionSpline[0] + 1f) / 2f) * (maxSplineStep - minSplineStep))) * 0.05f;
                    //_sampledActionSpline[nValuesPerSegment] = (minSplineStep + (int)(((_sampledActionSpline[nValuesPerSegment] + 1f) / 2f) * (maxSplineStep - minSplineStep))) * 0.05f;

                    if (agentParameters.onDemandDecision)
                        agentParameters.numberOfActionsBetweenDecisions = spline_steps_needed;// (int)(_sampledActionSpline[0] / Time.fixedDeltaTime);
                }
            }

            if (useSpline)
            {
                GenerateControlValues(counter_spline_not_updated);
                //if (currentDecisionStep % agentParameters.numberOfActionsBetweenDecisions == 0)
                //{
                ApplyControl(interpolatedControl);
                //}
            }

            ApplyConnectDisconnect();

            if (agentParameters.onDemandDecision)
                RequestAction();
            // increasing time
            IncrementDecisionTimer();
        }

        return;
    }

    void ApplyConnectDisconnect()
    {
        for (int b = 3; b >= 0; b--)
        {
            float dis = float.MaxValue;
            if (connectBodyParts[b].target_hold_id >= 0)
                dis = (holdPoses[connectBodyParts[b].target_hold_id].localPosition - connectBodyParts[b].GetRigidBody().transform.localPosition).magnitude;
            if (b <= 1)
            {
                if (connectBodyParts[2].IsConnected || connectBodyParts[3].IsConnected)
                {
                    connectBodyParts[b].connectBodyPart(dis);
                }
                else
                {
                    connectBodyParts[b].disconnectBodyPart();
                }
            }
            else
            {
                connectBodyParts[b].connectBodyPart(dis);
            }
        }
        return;
    }

    int count_decisions_per_episode = 0;
    void IncrementDecisionTimer()
    {
        counter_agent_running++;

        if (counter_agent_running % agentParameters.numberOfActionsBetweenDecisions == 0)
        {
            SaveInTrajectory();
        }

        if (!useSpline)
        {
            if (currentDecisionStep == agentParameters.numberOfActionsBetweenDecisions ||
                agentParameters.numberOfActionsBetweenDecisions == 1)
            {
                currentDecisionStep = 1;
                isNewDecisionStep = true;
                count_decisions_per_episode++;
            }
            else
            {
                currentDecisionStep++;
                isNewDecisionStep = false;
            }
        }
        else
        {
            if (currentDecisionStep == spline_steps_needed)
            {
                currentDecisionStep = 0;
                spline_steps_needed = 0;
            }
            else
            {
                currentDecisionStep++;
            }

            if (currentDecisionStep % agentParameters.numberOfActionsBetweenDecisions == 0)
            {
                if (agentParameters.onDemandDecision)
                    RequestDecision();
                isNewDecisionStep = true;
            }
            else
            {
                isNewDecisionStep = false;
            }
        }

        return;
    }

    void RandomizeScene()
    {
        if (HighLevelGuide.flag_test_performance)
            return;
        // min-dis = cDisBtwHolds - 2 * max_r, max-dis = cDisBtwHolds + 2 * max_r
        float cDisBtwHolds = 2.0f;
        float max_r = 1.0f;

        float cHoldSize = 0.5f;

        for (int h = 0; h < holdPoses.Length; h++)
        {
            HighLevelGuide.UpdateInitHoldPosition(h, cDisBtwHolds);
        }
        // randomize scene
        for (int h = 0; h < holdPoses.Length; h++)
        {
            Vector3 nPos = HighLevelGuide.GetInitHoldPosition(h)
                + Random.Range(0.1f, max_r) * (Quaternion.Euler(0, 0, Random.Range(0, 360)) * Vector3.right);

            holdPoses[h].transform.localPosition = HighLevelGuide.GetRestrictedHoldPos(nPos);
            
            holdPoses[h].transform.localScale = cHoldSize * Vector3.one;
            holdPoses[h].GetComponent<HoldInfo>().obstacleIndex = -1;
        }

        return;
    }
        
    int current_policy_update = 0;
    Vector2Int fromInitIndexPerformanceNodeIds = new Vector2Int(-1, -1);
    public override void AgentReset()
    {
        if (!useAgentInterface)
        {
            for (int b = 0; b < 4; b++)
            {
                planned_target_ids[b] = -1;
            }

            bool flag_update = false;
            int[] bodyIndex = { -1, -1 , -1, -1};
            if (fatherNodeID >= 0)
            {
                for (int b = 0; b < 4; b++)
                {
                    if (HighLevelGuide.graphNodes[fatherNodeID].holdIds[b] != target_hold_ids[b] && target_hold_ids[b] > -1)
                    {
                        flag_update = true;
                        bodyIndex[b] = b;
                    }
                }
            }
            
            if (flag_update)
            {
                int state_id = HighLevelGuide.graphNodes[fatherNodeID].humanoidStateIds[trajectoryNum];
                HumanoidClimberAcademy.HumanoidBodyState _cState = HighLevelGuide.savedStates[state_id];
                Vector3 ref_pose = HumanoidClimberAgent.localAgentScaling * _cState.pos[0] + new Vector3(0.01398927f, -6.0f, -1.5f);
                Quaternion ref_q = _cState.rot[0];
                Vector3 nPos = Quaternion.Inverse(ref_q) * ((targetPlannedHoldPos - ref_pose) / localAgentScaling);
                //Vector3 nPos = jdController.bones[0].rb.transform.InverseTransformPoint(targetPlannedHoldPos);
                HighLevelGuide.AddSitePos(nPos.x, nPos.y, GetCumulativeReward(), succeeded_transition, bodyIndex);
                HighLevelGuide.AddTransitionReward(GetCumulativeReward(), count_decisions_per_episode, num_spline_done, isTrajectoryForked, succeeded_transition, bodyIndex);
                
                HighLevelGuide.CompleteTestPerformanceNode(agent_state_ids, current_step, trajectoryNum, fromInitIndexPerformanceNodeIds, succeeded_transition);
            }
            
            count_decisions_per_episode = 0;

            SaveInTrajectory();
    
            if (flag_update)
            {
                if (HighLevelGuide.numUpdatedPolicy > HighLevelGuide.graphNodes[fatherNodeID].updated_node_num)
                {
                    HighLevelGuide.graphNodes[fatherNodeID].updated_node_num = HighLevelGuide.numUpdatedPolicy;
                    HighLevelGuide.graphNodes[fatherNodeID].preNodeRLValue = HighLevelGuide.graphNodes[fatherNodeID].GetNodeValue();
                    HighLevelGuide.graphNodes[fatherNodeID].NodeRLValue = 0;
                    HighLevelGuide.graphNodes[fatherNodeID].NumTriedTransitions = 0;
                }
                if (succeeded_transition)
                {
                    HighLevelGuide.graphNodes[fatherNodeID].NodeRLValue += 1.0f;
                }
                else
                {
                    HighLevelGuide.graphNodes[fatherNodeID].NodeRLValue -= 1.0f;
                }
                HighLevelGuide.graphNodes[fatherNodeID].NumTriedTransitions++;

                HighLevelGuide.UpdateSeenStance(fatherNodeID);

                HighLevelGuide.AddStanceTarget(agent_state_ids, current_step, target_hold_ids, current_policy_update >= HighLevelGuide.numUpdatedPolicy, GetCumulativeReward());
            }

            HighLevelGuide.AddSeenStance(agent_state_ids[current_step], current_policy_update >= HighLevelGuide.numUpdatedPolicy);

            if (current_step > 3000 || reset_agent_state_needed || !reset_each_transition)
            {
                reset_agent_state_needed = false;

                HighLevelGuide.AddPathReward(HighLevelGuide.savedStates[agent_state_ids[current_step]].current_accumulative_reward, isTrajectoryForked);

                not_moved_limbs.Clear();

                current_step = -1;
                starting_step = 0;
                starting_accumulative_reward = 0;

                agent_tree_task.Clear();
                task_planned_num = 0;
                task_failed_num = 0;

                RandomizeScene();

                // chosen init state to load for random training
                Vector2Int _stateID_forkedStatus = HighLevelGuide.GetRandomState(trajectoryNum, ref planned_target_ids, ref fromInitIndexPerformanceNodeIds);

                LoadState(_stateID_forkedStatus[0]);

                isTrajectoryForked = _stateID_forkedStatus[1] == 1;
                
                SaveInTrajectory();

                isTaskOnLadderClimbing = true;
                current_idx_limb_planned = HighLevelGuide.graphNodes[fatherNodeID].planning_index_on_ladder;
            }
            else
            {
                starting_accumulative_reward += GetCumulativeReward();
            }

            counter_agent_running = 0;
            num_spline_done = 0;
            task_counter_step = 0;

            isNewDecisionStep = true;
            if (useSpline)
            {
                currentDecisionStep = 0;
                spline_steps_needed = 0;
            }
            else
            {
                currentDecisionStep = 1;
            }

            for (int b = 0; b < 4; b++)
            {
                init_hold_ids[b] = connectBodyParts[b].cHoldId;
                connectBodyParts[b].target_hold_id = connectBodyParts[b].cHoldId;
                target_hold_ids[b] = connectBodyParts[b].cHoldId;
            }

            fatherNodeID = HighLevelGuide.ReturnNodeId(init_hold_ids);
            if (HighLevelGuide.ReturnNodeId(init_hold_ids) == -1)
            {
                fatherNodeID = HighLevelGuide.AddNode(fatherNodeID, init_hold_ids, trajectoryNum, true);
            }
            if (!agent_tree_task.Contains(HumanoidClimberAcademy.StanceToKey(HighLevelGuide.graphNodes[fatherNodeID].holdIds)))
                agent_tree_task.Add(HumanoidClimberAcademy.StanceToKey(HighLevelGuide.graphNodes[fatherNodeID].holdIds));

            bool isTargetSet = false;
            int max_target_stance_sample = HumanoidClimberAcademy.current_training_type != HumanoidClimberAcademy.TrainingType.RandomTraining ? 100 : 1;

            HighLevelGuide.GetNextPlannedTargetHoldIds(ref fromInitIndexPerformanceNodeIds, ref planned_target_ids);
            if (HumanoidClimberAcademy.Tools.GetNumConnectedLimbs(planned_target_ids) != 0)
            {
                max_target_stance_sample = 1;
            }

            for (int s = 0; s < max_target_stance_sample && !isTargetSet; s++)
            {
                SetNextTargetHoldIds(true);
                bool isHandTargetSet = false;
                bool isFeetTargetSet = false;
                for (int i = 0; i < 4 && !isTargetSet; i++)
                {
                    if (connectBodyParts[i].cHoldId != connectBodyParts[i].target_hold_id)
                    {
                        if (i == 0 || i == 1)
                        {
                            isFeetTargetSet = true;
                        }
                        if (i == 2 || i == 3)
                        {
                            isHandTargetSet = true;
                        }
                    }
                }

                if (isFeetTargetSet || isHandTargetSet)
                {
                    isTargetSet = true;
                }
                if (HighLevelGuide.IsHandSampleNeeded() && !isHandTargetSet)
                {
                    isTargetSet = false;
                }
            }

            current_policy_update = HighLevelGuide.numUpdatedPolicy;

            if (useObstacles)
            {
                float defaultZ = 3.75f + 0.125F * (HighLevelGuide.loggedStep / (5e6f));
                if (defaultZ > 3.875f)
                    defaultZ = 3.875f;
                for (int i = 0; i < 4; i++)
                {
                    if (init_hold_ids[i] != target_hold_ids[i] && target_hold_ids[i] >= 0)
                    {
                        Vector3 p = HighLevelGuide.GetInitHoldPosition(target_hold_ids[i]);
                        Obstacles[4].transform.localPosition = new Vector3(p.x + Random.Range(-1.0f, 1.0f), p.y + Random.Range(-1.0f, 1.0f), defaultZ);
                        Obstacles[4].transform.rotation = Quaternion.Euler(0, 0, Random.Range(0.0f, 359.9f));
                        Obstacles[4].transform.localScale = new Vector3(Random.Range(0.5f, 1.0f), Random.Range(0.5f, 1.0f), Random.Range(0.5f, 1.0f));
                    }
                }
            }
            
        }
        else
        {
            current_step = 0;
            starting_step = 0;

            agent_tree_task.Clear();
            counter_agent_running = 0;
            num_spline_done = 0;
            task_counter_step = 0;
            task_planned_num = 0;
            task_failed_num = 0;

            isNewDecisionStep = true;
            if (useSpline)
            {
                currentDecisionStep = 0;
                spline_steps_needed = 0;
            }
            else
            {
                currentDecisionStep = 1;
            }

            current_count_limb_movment = 0;
            isTrajectoryForked = false;
            fatherNodeID = -1;
            isTaskOnLadderClimbing = false;
            current_idx_limb_planned = -1;

            for (int b = 0; b < 4; b++)
            {
                init_hold_ids[b] = connectBodyParts[b].cHoldId;
            }
        }
        return;
    }

    void SaveInTrajectory()
    {
        int[] cHoldIDs = new int[] { connectBodyParts[0].cHoldId,
                                     connectBodyParts[1].cHoldId,
                                     connectBodyParts[2].cHoldId,
                                     connectBodyParts[3].cHoldId};
        
        current_step++;
        while (current_step >= agent_state_ids.Count)
        {
            agent_state_ids.Add(-1);
        }

        if (agent_state_ids[current_step] == -1)
        {
            agent_state_ids[current_step] = HighLevelGuide.GetNextFreeState(jdController.getBonesCount(), spline.Count);
        }

        int _stateID = agent_state_ids[current_step];

        SaveState(_stateID);

        if (HighLevelGuide.ReturnNodeId(cHoldIDs) == -1)
        {
            HighLevelGuide.AddNode(fatherNodeID, cHoldIDs, trajectoryNum, true);
        }
        return;
    }

    public void SaveState(int _stateID)
    {
        if (_stateID < 0)
        {
            return;
        }
        HumanoidClimberAcademy.HumanoidBodyState _cState = HighLevelGuide.savedStates[_stateID];

        _cState.current_accumulative_reward = this.GetCumulativeReward();
        if (reset_each_transition)
            _cState.current_accumulative_reward += starting_accumulative_reward;
        _cState.current_step = starting_step + current_step;

        if (_cState.current_step < 0)
        {
            _cState.current_step = 0;
        }

        if (useSpline)
        {
            for (int i = 0; i < nActionSize; i++)
            {
                _cState.spline_init_values[i][0] = spline[i].currentValue;
                _cState.spline_init_values[i][1] = spline[i].currentDerivativeValue;
            }
        }

        for (int b = 0; b < 4; b++)
        {
            _cState.current_hold_ids[b] = connectBodyParts[b].cHoldId;
            int obs_index = 4;
            if (connectBodyParts[b].cHoldId >= 0)
            {
                _cState.current_hold_pos[b] = holdPoses[connectBodyParts[b].cHoldId].localPosition;
                obs_index = holdPoses[connectBodyParts[b].cHoldId].GetComponent<HoldInfo>().obstacleIndex;

                if (obs_index < 0)
                {
                    float minDis = float.MaxValue;
                    int idx = -1;
                    for (int o = 0; o < 5; o++)
                    {
                        float dis = (_cState.current_hold_pos[b] - Obstacles[o].transform.localPosition).magnitude;
                        if (dis < minDis)
                        {
                            idx = o;
                            dis = minDis;
                        }
                    }
                    obs_index = idx;
                }
            }
            if (obs_index >= 0)
            {
                _cState.obstacleHoldPoses[b] = Obstacles[obs_index].transform.localPosition;
                _cState.obstacleHoldRots[b] = Obstacles[obs_index].transform.rotation;
                _cState.obstacleHoldScale[b] = Obstacles[obs_index].transform.localScale;
            }

            _cState.connectorTouchingGround[b] = connectBodyParts[b].groundContact.touchingGround;
            _cState.connectorTouchingWall[b] = connectBodyParts[b].wallContact.touchingWall;
            _cState.connectorTouchingHold[b] = connectBodyParts[b].holdContact.touchingHold;

            _cState.connectorPos[b] = connectBodyParts[b].GetRigidBody().transform.localPosition;
            _cState.connectorRot[b] = connectBodyParts[b].GetRigidBody().transform.localRotation;
            _cState.connectorVel[b] = connectBodyParts[b].GetRigidBody().velocity;
            _cState.connectorAVel[b] = connectBodyParts[b].GetRigidBody().angularVelocity;
        }

        for (int i = 0; i < jdController.bones.Count; i++)
        {
            _cState.pos[i] = jdController.bones[i].rb.transform.localPosition;
            _cState.rot[i] = jdController.bones[i].rb.transform.localRotation;
            _cState.vel[i] = jdController.bones[i].rb.velocity;
            _cState.aVel[i] = jdController.bones[i].rb.angularVelocity;

            _cState.touchingGround[i] = jdController.bones[i].groundContact.touchingGround;
            _cState.touchingWall[i] = jdController.bones[i].wallContact.touchingWall;

            _cState.touchingHold[i] = jdController.bones[i].holdContact.touchingHold;
            _cState.touchingHoldId[i] = jdController.bones[i].holdContact.hold_id;

            _cState.touchingTarget[i] = jdController.bones[i].targetContact.touchingTarget;

        }

        for (int j = 0; j < jdController.numControlDOFs(); j++)
        {
            _cState.currentTargetAngles[j] = jdController.cTargetAngles[j];
        }
        return;
    }

    public void LoadState(int _stateID)
    {
        if (HighLevelGuide.flag_test_performance)
            flag_random_shift = false;

        HumanoidClimberAcademy.HumanoidBodyState _cState = HighLevelGuide.savedStates[_stateID];

        if (useSpline)
        {
            for (int i = 0; i < nActionSize; i++)
            {
                spline[i].setState(_cState.spline_init_values[i][0], _cState.spline_init_values[i][1]);
                spline[i].linearMix = 0.0f;
            }
        }

        int[] current_hold_ids = { _cState.current_hold_ids[0],
                                   _cState.current_hold_ids[1],
                                   _cState.current_hold_ids[2],
                                   _cState.current_hold_ids[3]};
        Vector3[] current_hold_pos = { _cState.current_hold_pos[0],
                                       _cState.current_hold_pos[1],
                                       _cState.current_hold_pos[2],
                                       _cState.current_hold_pos[3]};
        Vector3 hip_displacement = new Vector3(0, 0, 0);
        if (flag_random_shift)
        {
            bool handConnected = _cState.current_hold_ids[2] >= 0 && _cState.current_hold_ids[3] >= 0;
            bool legConnected = _cState.current_hold_ids[0] >= 0 && _cState.current_hold_ids[1] >= 0;
            if (handConnected && legConnected)
            {
                if (HumanoidClimberAcademy.current_training_type == HumanoidClimberAcademy.TrainingType.UniformAroundStateTraining)
                {
                    hip_displacement = HumanoidClimberAgent.localAgentScaling * (new Vector3(0.0f, Random.Range(1.25f, 2.0f) - _cState.pos[0].y, 0.0f));
                    for (int i = 0; i < 4; i++)
                    {
                        if (_cState.current_hold_ids[i] >= 0)
                        {
                            current_hold_pos[i] += hip_displacement;
                        }
                    }
                }
                else
                {
                    Vector2Int row_minus_plus = new Vector2Int(int.MaxValue, int.MaxValue);
                    Vector2Int col_minus_plus = new Vector2Int(int.MaxValue, int.MaxValue);
                    int[] row_cur_ids = { -1, -1, -1, -1 };
                    int[] col_cur_ids = { -1, -1, -1, -1 };
                    for (int i = 0; i < 4; i++)
                    {
                        if (_cState.current_hold_ids[i] >= 0)
                        {
                            int row = (int)(_cState.current_hold_ids[i] / 4);
                            row_cur_ids[i] = row;
                            int plus_row_val = Mathf.Abs(3 - row);
                            int minus_row_val = Mathf.Abs(0 - row);
                            row_minus_plus[1] = Mathf.Min(row_minus_plus[1], plus_row_val);
                            row_minus_plus[0] = Mathf.Min(row_minus_plus[0], minus_row_val);

                            int col = (int)(_cState.current_hold_ids[i] % 4);
                            col_cur_ids[i] = col;
                            int plus_col_val = Mathf.Abs(3 - col);
                            int minus_col_val = Mathf.Abs(0 - col);
                            col_minus_plus[1] = Mathf.Min(col_minus_plus[1], plus_col_val);
                            col_minus_plus[0] = Mathf.Min(col_minus_plus[0], minus_col_val);
                        }
                    }

                    int row_displacement = 0;
                    int col_displacement = 0;
                    
                    float rnd_row = Random.Range(0.0f, 1.0f);
                    float row_dis_val = rnd_row * (row_minus_plus[0] + 1 + row_minus_plus[1] + 1);
                    if (row_dis_val < row_minus_plus[0] + 1)
                    {
                        row_displacement = (int)(row_dis_val);
                        if (row_displacement < 0) row_displacement = 0;
                        if (row_displacement >= row_minus_plus[0] + 1) row_displacement = row_minus_plus[0];
                        row_displacement = -row_displacement;
                    }
                    else
                    {
                        row_displacement = (int)(row_dis_val - (row_minus_plus[0] + 1));
                        if (row_displacement < 0) row_displacement = 0;
                        if (row_displacement >= row_minus_plus[1] + 1) row_displacement = row_minus_plus[1];
                    }

                    float rnd_col = Random.Range(0.0f, 1.0f);
                    float col_dis_val = rnd_col * (col_minus_plus[0] + 1 + col_minus_plus[1] + 1);
                    if (col_dis_val < col_minus_plus[0] + 1)
                    {
                        col_displacement = (int)(col_dis_val);
                        if (col_displacement < 0) col_displacement = 0;
                        if (col_displacement >= col_minus_plus[0] + 1) col_displacement = col_minus_plus[0];
                        col_displacement = -col_displacement;
                    }
                    else
                    {
                        col_displacement = (int)(col_dis_val - (col_minus_plus[0] + 1));
                        if (col_displacement < 0) col_displacement = 0;
                        if (col_displacement >= col_minus_plus[1] + 1) col_displacement = col_minus_plus[1];
                    }
                    
                    int counter_avg = 0;
                    for (int i = 0; i < 4; i++)
                    {
                        if (_cState.current_hold_ids[i] >= 0)
                        {
                            counter_avg++;
                            int nRow = row_cur_ids[i] + row_displacement;
                            int nCol = col_cur_ids[i] + col_displacement;
                            current_hold_ids[i] = 4 * nRow + nCol;

                            Vector3 pos_displcement = HighLevelGuide.GetInitHoldPosition(current_hold_ids[i]) - HighLevelGuide.GetInitHoldPosition(_cState.current_hold_ids[i]);
                            current_hold_pos[i] += pos_displcement;
                            hip_displacement += pos_displcement;
                        }
                    }
                    if (counter_avg > 0)
                        hip_displacement /= counter_avg;
                }
            }
        }
        // loading the position of saved holds for the current state of the climber
        for (int b = 0; b < 4; b++)
        {
            if (useObstacles)
            {
                Obstacles[b].transform.localPosition = _cState.obstacleHoldPoses[b] + hip_displacement;
                Obstacles[b].transform.rotation = _cState.obstacleHoldRots[b];
                Obstacles[b].transform.localScale = _cState.obstacleHoldScale[b];
            }
            if (current_hold_ids[b] >= 0)
                holdPoses[current_hold_ids[b]].localPosition = current_hold_pos[b];

            connectBodyParts[b].target_hold_id = -1;
            connectBodyParts[b].disconnectBodyPart();

            connectBodyParts[b].groundContact.touchingGround = _cState.connectorTouchingGround[b];
            connectBodyParts[b].wallContact.touchingWall = _cState.connectorTouchingWall[b];
            
            connectBodyParts[b].holdContact.touchingHold = _cState.connectorTouchingHold[b];
        }

        for (int b = 0; b < 4; b++)
        {
            connectBodyParts[b].GetRigidBody().transform.localPosition = _cState.connectorPos[b] + hip_displacement;
            connectBodyParts[b].GetRigidBody().transform.localRotation = _cState.connectorRot[b];
            connectBodyParts[b].GetRigidBody().velocity = _cState.connectorVel[b];
            connectBodyParts[b].GetRigidBody().angularVelocity = _cState.connectorAVel[b];

            connectBodyParts[b].target_hold_id = current_hold_ids[b];
            if (current_hold_ids[b] >= 0)
            {
                connectBodyParts[b].holdContact.touchingHold = true;
                connectBodyParts[b].holdContact.hold_id = current_hold_ids[b];
            }
        }

        for (int i = 0; i < jdController.bones.Count; i++)
        {
            // only shift the hip in the hierarchy
            if (i == 0)
            {
                jdController.bones[i].rb.transform.localPosition = _cState.pos[i] + hip_displacement / HumanoidClimberAgent.localAgentScaling;
            }
            else
                jdController.bones[i].rb.transform.localPosition = _cState.pos[i];
            
            jdController.bones[i].rb.transform.localRotation = _cState.rot[i];
            jdController.bones[i].rb.velocity = _cState.vel[i];
            jdController.bones[i].rb.angularVelocity = _cState.aVel[i];

            jdController.bones[i].groundContact.touchingGround = _cState.touchingGround[i];
            jdController.bones[i].wallContact.touchingWall = _cState.touchingWall[i];

            jdController.bones[i].holdContact.touchingHold = _cState.touchingHold[i];
            jdController.bones[i].holdContact.hold_id = _cState.touchingHoldId[i];

            jdController.bones[i].targetContact.touchingTarget = _cState.touchingTarget[i];
        }

        jdController.driveToPose(_cState.currentTargetAngles);
        
        ApplyConnectDisconnect();

        starting_step = _cState.current_step;
        //if (reset_each_transition)
        //{
        //    starting_accumulative_reward = _cState.current_accumulative_reward;
        //}
        return;
    }

    Vector3 GetCOM()
    {
        return jdController.COM();
    }

    Vector3 GetTargetedHoldPos(bool getGlobal)
    {
        Vector3 tPos = new Vector3(0f, 0f, 0f);
        int c = 0;
        for (int b = 0; b < 4; b++)
        {
            if (connectBodyParts[b].target_hold_id != init_hold_ids[b])
            {
                if (connectBodyParts[b].target_hold_id >= 0 && connectBodyParts[b].target_hold_id < holdPoses.Length)
                {
                    if (getGlobal)
                    {
                        tPos += holdPoses[connectBodyParts[b].target_hold_id].position;
                    }
                    else
                    {
                        tPos += holdPoses[connectBodyParts[b].target_hold_id].localPosition;
                    }
                    c++;
                }
            }
        }

        return tPos / (c + 1e-6f);
    }

    private void OnApplicationQuit()
    {
        if (trajectoryNum != 0)
            return;
        if (user_trajectory_points.Count > 0 && HighLevelGuide.flag_test_performance)
        {
            StreamReader mRouteNumFile = new StreamReader("ClimberRouteInfo\\RouteNum.txt");
            int route_num = int.Parse(mRouteNumFile.ReadLine());
            mRouteNumFile.Close();
            StreamWriter userDesiredStanceFile = new StreamWriter("ClimberRouteInfo\\RouteStance" + route_num.ToString() + ".txt");

            for (int i = 0; i < user_trajectory_points.Count; i++)
            {
                string s = user_trajectory_points[i].target_ids[0].ToString() + ","
                         + user_trajectory_points[i].target_ids[1].ToString() + ","
                         + user_trajectory_points[i].target_ids[2].ToString() + ","
                         + user_trajectory_points[i].target_ids[3].ToString();
                userDesiredStanceFile.WriteLine(s);
            }

            userDesiredStanceFile.Flush();
            userDesiredStanceFile.Close();
        }
    }
}
