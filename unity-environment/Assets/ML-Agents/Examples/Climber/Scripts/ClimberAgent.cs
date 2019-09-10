using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class ClimberAgent : Agent
{
    static int m_agent_counter = 0;
    public class EndBodyPart
    {
        public bool IsConnected
        {
            get { return isConnected; }
        }

        public EndBodyPart(Transform bodyPartInfo, ClimberAgent _p)
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

        public void connectBodyPart()
        {
            if (holdContact.touchingHold && holdContact.hold_id == target_hold_id)
            {
                if (cHoldId != target_hold_id)
                {
                    _locker_body.isKinematic = true;

                    isConnected = true;
                    cHoldId = target_hold_id;
                }
            }
            else if (cHoldId != target_hold_id && cHoldId >= 0)
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
    public HighLevelPlanner TaskPlanner;
    public ClimberAcademy HighLevelGuide;
    public int trajectoryNum = -1;

    [Header("Holds in Environment")]
    public Transform[] holds_pos;

    [Header("Body Parts")]
    public Transform hips;
    public Transform chest;
    public Transform spine;
    public Transform head;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform armL;
    public Transform forearmL;
    public Transform handL;
    public Transform armR;
    public Transform forearmR;
    public Transform handR;

    [Header("End Points Connectors")]
    public Transform RHConnector;
    public Transform LHConnector;
    public Transform RLConnector;
    public Transform LLConnector;

    [HideInInspector] public JointDriveController jdController;
    bool isNewDecisionStep;
    int currentDecisionStep;
    List<EndBodyPart> connectBodyParts = new List<EndBodyPart>();
    float[] sampleVectorAction;

    //user control agent boolean params
    [HideInInspector] public const bool useAgentInterface = false; // where the task instruction are coming from?
    [HideInInspector] const bool reset_each_transition = true;
    [HideInInspector] public const bool flag_test_performance = false && !useAgentInterface;
    [HideInInspector] public const bool flag_random_shift = !useAgentInterface && true; // where the task instruction are coming from?
    [HideInInspector] const bool useSpline = true;

    // for test performance
    [HideInInspector] const int minConnectedLimbsInitHolds = 4;
    Vector3 targetPlannedHoldPos = new Vector3(0f, 0f, 0f);
    bool succeeded_transition = false;

    // general variables for each task
    int task_planned_num = 0;
    int task_failed_num = 0;
    int task_counter_step = 0;
    bool isTaskCompleted = false;
    
    // lader climbing
    enum LadderClimbingTaskType { GoUp = 0, GoRight = 1, GoLeft = 2, GoDown = 3};
    LadderClimbingTaskType currentTastType = LadderClimbingTaskType.GoUp;
    List<int> not_moved_limbs = new List<int>();
    int current_idx_limb_planned = 0;

    //////////////////////////////////////begin for spline /////////////////////////////////////
    // if useSpline = true and useLinearSpline = false, then make observation size = 325, action size = 80 in unity
    // if useSpline = true and useLinearSpline = false, then make observation size = 325, action size = 40 in unity
    // if useSpline = false, then make observation size = 247, action size = 39 in unity
    
    const int nActionSize = 39;
    const bool useLinearSpline = false;
    const int nControlPoints = useLinearSpline ? 1 : 2;
    public List<RecursiveTCBSpline> spline = new List<RecursiveTCBSpline>();
    float[] interpolatedControl;
    int minSplineStep = 1, maxSplineStep = 10;
    int max_spline_used = 1000;
    float[] _sampledActionSpline;
    int num_spline_done = 0;
    int spline_steps_needed = 0;
    int counter_spline_not_updated = 0;
    List<Vector2> MinMaxControlValues = new List<Vector2>();
                                   ////////////////////////////////////////end for spline /////////////////////////////////////

    // init and target hold ids
    int[] init_hold_ids = { -1, -1, -1, -1};
    int[] target_hold_ids = { -1, -1, -1, -1 };
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
        user_trajectory_points[user_trajectory_points.Count - 1].cStateID = HighLevelGuide.GetNextFreeState(jdController.bodyPartsList.Count, spline.Count);

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
                footL.gameObject.GetComponent<Renderer>().material.color = _color;
                break;
            case 1:
                RLConnector.gameObject.GetComponent<Renderer>().material.color = _color;
                footR.gameObject.GetComponent<Renderer>().material.color = _color;
                break;
            case 2:
                LHConnector.gameObject.GetComponent<Renderer>().material.color = _color;
                handL.gameObject.GetComponent<Renderer>().material.color = _color;
                break;
            case 3:
                RHConnector.gameObject.GetComponent<Renderer>().material.color = _color;
                handR.gameObject.GetComponent<Renderer>().material.color = _color;
                break;
        }
    }

    void SetColorToHolds(int hold_id, Color _color)
    {
        if (hold_id > -1)
            holds_pos[hold_id].gameObject.GetComponent<Renderer>().material.color = _color;
        return;
    }

    private void Update()
    {
        if (!useAgentInterface)
            return;
        
        if (trajectoryNum != 0)
            return;

        Camera.main.GetComponent<CameraFollow>().flagFollowAgent = !useAgentInterface;

        Vector3 nCameraPos = lookAtPos + camera_pos[0] * (new Vector3(Mathf.Cos(camera_pos[1]), Mathf.Sin(camera_pos[2]), Mathf.Sin(camera_pos[1]))).normalized;
        nCameraPos.y = Mathf.Max(0f, nCameraPos.y);

        Camera.main.transform.position = nCameraPos;
        Camera.main.transform.LookAt(lookAtPos, Vector3.up);

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

        for (int h = 0; h < holds_pos.Length; h++)
        {
            SetColorToHolds(h, Color.white);
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
                        holds_pos[user_trajectory_points[user_trajectory_points.Count - 1].target_ids[b]].position);
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

            current_count_limb_movment = ClimberAcademy.Tools.GetDiffBtwSetASetB(target_hold_ids, init_hold_ids) - 1;

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
        if (ClimberAcademy.current_training_type == ClimberAcademy.TrainingType.GraphTreeTraining)
        {
            currentTastType = LadderClimbingTaskType.GoUp;
            return;
        }

        if (ClimberAcademy.current_training_type != ClimberAcademy.TrainingType.LadderClimbing)
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
        if (!flag_test_performance)
        {
            if (!(ClimberAcademy.current_training_type == ClimberAcademy.TrainingType.GraphTreeTraining
                || ClimberAcademy.current_training_type == ClimberAcademy.TrainingType.LadderClimbing))
            {
                return;
            }
        }
        if (current_idx_limb_planned >= 4 || (not_moved_limbs.Count == 0 && ClimberAcademy.current_training_type == ClimberAcademy.TrainingType.LadderClimbing))
        {
            SetRandomLadderClimbingTask();
            current_idx_limb_planned = 0;
        }
        int n_limb_id = -1;
        if (ClimberAcademy.current_training_type == ClimberAcademy.TrainingType.LadderClimbing)
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

            if (flag_test_performance)
            {
                if (ClimberAcademy.Tools.GetNumConnectedLimbs(cHoldIDs) < minConnectedLimbsInitHolds && agent_state_ids.Count == 0)
                {
                    PlanForLadderClimbing();
                    return;
                }
            }
            
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

            current_count_limb_movment = Random.Range((int)ClimberAcademy.min_count_limb_movement - 1, (int)ClimberAcademy.max_count_limb_movement);
            // update from-to stance nodes on the stance graph
            fatherNodeID = HighLevelGuide.ReturnNodeId(init_hold_ids);

            HighLevelGuide.UpdateHumanoidState(fatherNodeID, trajectoryNum);
            HighLevelGuide.CompleteNode(fatherNodeID, current_count_limb_movment);

            if (ClimberAcademy.Tools.GetNumConnectedLimbs(planned_target_ids) == 0)
            {
                if (isTrajectoryForked && (ClimberAcademy.current_training_type == ClimberAcademy.TrainingType.GraphTreeTraining
                    || ClimberAcademy.current_training_type == ClimberAcademy.TrainingType.LadderClimbing))
                {
                    float randomly_do_goup = 0.0f;
                    if (ClimberAcademy.current_training_type == ClimberAcademy.TrainingType.GraphTreeTraining)
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

                    if ((ClimberAcademy.current_training_type == ClimberAcademy.TrainingType.LadderClimbing || randomly_do_goup >= 0.85f) && isTaskOnLadderClimbing)
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

            current_count_limb_movment = ClimberAcademy.Tools.GetDiffBtwSetASetB(target_hold_ids, init_hold_ids) - 1;

            if (!ClimberAcademy.IsValidTransition(init_hold_ids, target_hold_ids))
            {
                for (int b = 0; b < 4; b++)
                {
                    connectBodyParts[b].target_hold_id = connectBodyParts[b].cHoldId;
                }
                return;
            }

            // during training, we do not want to have repeated movements (we want more diverse movements)
            if (isTrajectoryForked)
            {
                ulong stanceNodeID = ClimberAcademy.StanceToKey(target_hold_ids);
                if (agent_tree_task.Contains(stanceNodeID))
                {
                    MyDoneFunc(false, true);
                }
                else
                {
                    agent_tree_task.Add(stanceNodeID);
                }
            }

        }

        return;
    }

    void ApplyControl(float[] vectorAction)
    {
        var bpDict = jdController.bodyPartsDict;
        int i = -1;

        if (!useSpline)
        {
            bpDict[chest].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[spine].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);

            bpDict[thighL].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[thighR].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[shinL].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[shinR].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[footR].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[footL].SetJointTargetRotation(vectorAction[++i], 0, 0);


            bpDict[armL].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[armR].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[forearmL].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[forearmR].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[head].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);

            //update joint strength settings
            bpDict[chest].SetJointStrength(vectorAction[++i]);
            bpDict[spine].SetJointStrength(vectorAction[++i]);
            bpDict[head].SetJointStrength(vectorAction[++i]);
            bpDict[thighL].SetJointStrength(vectorAction[++i]);
            bpDict[shinL].SetJointStrength(vectorAction[++i]);
            bpDict[footL].SetJointStrength(vectorAction[++i]);
            bpDict[thighR].SetJointStrength(vectorAction[++i]);
            bpDict[shinR].SetJointStrength(vectorAction[++i]);
            bpDict[footR].SetJointStrength(vectorAction[++i]);
            bpDict[armL].SetJointStrength(vectorAction[++i]);
            bpDict[forearmL].SetJointStrength(vectorAction[++i]);
            bpDict[armR].SetJointStrength(vectorAction[++i]);
            bpDict[forearmR].SetJointStrength(vectorAction[++i]);
        }
        else
        {
            bpDict[chest].SetJointTargetRotationTrueVal(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[spine].SetJointTargetRotationTrueVal(vectorAction[++i], vectorAction[++i], vectorAction[++i]);

            bpDict[thighL].SetJointTargetRotationTrueVal(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[thighR].SetJointTargetRotationTrueVal(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[shinL].SetJointTargetRotationTrueVal(vectorAction[++i], 0, 0);
            bpDict[shinR].SetJointTargetRotationTrueVal(vectorAction[++i], 0, 0);
            bpDict[footR].SetJointTargetRotationTrueVal(vectorAction[++i], 0, 0);
            bpDict[footL].SetJointTargetRotationTrueVal(vectorAction[++i], 0, 0);


            bpDict[armL].SetJointTargetRotationTrueVal(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[armR].SetJointTargetRotationTrueVal(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[forearmL].SetJointTargetRotationTrueVal(vectorAction[++i], 0, 0);
            bpDict[forearmR].SetJointTargetRotationTrueVal(vectorAction[++i], 0, 0);
            bpDict[head].SetJointTargetRotationTrueVal(vectorAction[++i], vectorAction[++i], 0);

            //update joint strength settings
            bpDict[chest].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[spine].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[head].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[thighL].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[shinL].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[footL].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[thighR].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[shinR].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[footR].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[armL].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[forearmL].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[armR].SetJointStrengthTrueVal(vectorAction[++i]);
            bpDict[forearmR].SetJointStrengthTrueVal(vectorAction[++i]);
        }
    }

    void GenerateControlValues(int step)
    {
        // this function re-create spline given 'step' and (v,dv) at 'step' of spline
        // assuming we are at correct step of spline (i.e. spline v,dv are set properly)
        //, then given 'step' we can re-create same spline

        // maxStep calculated such that step is not violating segmentID
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

        float maxSpeed = 2.0f * Mathf.Rad2Deg * Mathf.PI * Time.fixedDeltaTime;// + 100f * (1e5f / (HighLevelGuide.loggedStep + 1e-6f))
        for (int i = 0; i < nActionSize; i++)
        {
            float s1 = sample[Mathf.Min(segmentIdx, nControlPoints - 1) * nValuesPerSegment + 1 + i], 
                  s2 = sample[Mathf.Min(segmentIdx + 1, nControlPoints - 1) * nValuesPerSegment + 1 + i];

            float p1 = ((s1 + 1.0f) / 2.0f) * (MinMaxControlValues[i][1] - MinMaxControlValues[i][0]) + MinMaxControlValues[i][0];
            float p2 = ((s2 + 1.0f) / 2.0f) * (MinMaxControlValues[i][1] - MinMaxControlValues[i][0]) + MinMaxControlValues[i][0];
            
            float cVal = spline[i].currentValue;
            spline[i].step(Time.fixedDeltaTime, p1, t1, p2, t2);

            if (i < nActionSize - 13)
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
        for (int i = 0; i < 13; i++)
        {
            float strength = vectorAction[vectorAction.Length - 1 - i];
            var rawVal = (strength + 1f) * 0.5f;

            dis += rawVal * rawVal;
        }

        return Mathf.Sqrt(dis);
    }

    void SetMinMaxXontrolValues()
    {
        var bpDict = jdController.bodyPartsDict;

        bpDict[chest].AddMinMaxTargetRotationValues(MinMaxControlValues, 3);
        bpDict[spine].AddMinMaxTargetRotationValues(MinMaxControlValues, 3);

        bpDict[thighL].AddMinMaxTargetRotationValues(MinMaxControlValues, 3);
        bpDict[thighR].AddMinMaxTargetRotationValues(MinMaxControlValues, 3);
        bpDict[shinL].AddMinMaxTargetRotationValues(MinMaxControlValues, 1);
        bpDict[shinR].AddMinMaxTargetRotationValues(MinMaxControlValues, 1);
        bpDict[footR].AddMinMaxTargetRotationValues(MinMaxControlValues, 1);
        bpDict[footL].AddMinMaxTargetRotationValues(MinMaxControlValues, 1);


        bpDict[armL].AddMinMaxTargetRotationValues(MinMaxControlValues, 3);
        bpDict[armR].AddMinMaxTargetRotationValues(MinMaxControlValues, 3);
        bpDict[forearmL].AddMinMaxTargetRotationValues(MinMaxControlValues, 1);
        bpDict[forearmR].AddMinMaxTargetRotationValues(MinMaxControlValues, 1);
        bpDict[head].AddMinMaxTargetRotationValues(MinMaxControlValues, 2);

        //update joint strength settings
        bpDict[chest].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[spine].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[head].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[thighL].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[shinL].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[footL].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[thighR].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[shinR].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[footR].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[armL].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[forearmL].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[armR].AddMinMaxStrengthValues(MinMaxControlValues);
        bpDict[forearmR].AddMinMaxStrengthValues(MinMaxControlValues);
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
        lookAtPos = hips.position;

        if (useAgentInterface)
        {
            agentParameters.maxStep = 0;
        }

        current_step = -1;

        while (HighLevelGuide._ControlRigs.Count < 16)
        {
            HighLevelGuide._ControlRigs.Add(null);
        }

        jdController = GetComponent<JointDriveController>();

        if (jdController.bodyPartsList.Count == 0)
        {
            jdController.SetupBodyPart(hips);
            jdController.SetupBodyPart(chest);
            jdController.SetupBodyPart(spine);
            jdController.SetupBodyPart(head);
            jdController.SetupBodyPart(thighL);
            jdController.SetupBodyPart(shinL);
            jdController.SetupBodyPart(footL);
            jdController.SetupBodyPart(thighR);
            jdController.SetupBodyPart(shinR);
            jdController.SetupBodyPart(footR);
            jdController.SetupBodyPart(armL);
            jdController.SetupBodyPart(forearmL);
            jdController.SetupBodyPart(handL);
            jdController.SetupBodyPart(armR);
            jdController.SetupBodyPart(forearmR);
            jdController.SetupBodyPart(handR);
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
            sampleVectorAction = new float[nControlPoints * (1 + nActionSize)];

            //agentParameters.onDemandDecision = true;
            SetMinMaxXontrolValues();
        }
        else
        {
            sampleVectorAction = new float[nActionSize];
        }
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
        return;
    }

    public override void CollectObservations()
    {
        //float noise_std = 1e-6f;

        jdController.GetCurrentJointForces();

        AddVectorObs(hips.forward);
        AddVectorObs(hips.up);

        foreach (var bodyPart in jdController.bodyPartsDict.Values)
        {
            var rb = bodyPart.rb;
            if (bodyPart.rb.transform != handL && bodyPart.rb.transform != handR)
            {
                AddVectorObs((bodyPart.groundContact.touchingGround ? 1 : 0)); // Is this bp touching the ground
                AddVectorObs((bodyPart.wallContact.touchingWall ? 1 : 0)); // Is this bp touching the wall
            }
            AddVectorObs(rb.velocity);
            AddVectorObs(rb.angularVelocity);
            if (bodyPart.rb.transform != hips)
            {
                Vector3 localPosRelToHips = hips.InverseTransformPoint(rb.position);
                if (float.IsNaN(localPosRelToHips.magnitude) || float.IsInfinity(localPosRelToHips.magnitude))
                {
                    localPosRelToHips = rb.position - hips.position;
                }
                AddVectorObs(localPosRelToHips);
            }

            if (bodyPart.rb.transform != hips && bodyPart.rb.transform != handL && bodyPart.rb.transform != handR &&
                bodyPart.rb.transform != footL && bodyPart.rb.transform != footR && bodyPart.rb.transform != head)
            {
                AddVectorObs(bodyPart.currentXNormalizedRot);
                AddVectorObs(bodyPart.currentYNormalizedRot);
                AddVectorObs(bodyPart.currentZNormalizedRot);
                AddVectorObs(bodyPart.currentStrength / (jdController.maxJointForceLimit + 1e-6f));
            }
        }

        // which body parts are moving
        for (int b = 0; b < 4; b++)
        {
            AddVectorObs(init_hold_ids[b] != target_hold_ids[b] ? 1f : 0f);
        }

        for (int b = 0; b < 4; b++)
        {
            AddVectorObs(connectBodyParts[b].cHoldId >= 0 ? 1f : 0f);
            AddVectorObs(connectBodyParts[b].target_hold_id >= 0 ? 1f : 0f);

            AddVectorObs(connectBodyParts[b].groundContact.touchingGround ? 1f : 0f); // Is this bp touching the ground
            AddVectorObs(connectBodyParts[b].wallContact.touchingWall ? 1f : 0f); // Is this bp touching the wall
            if (connectBodyParts[b].target_hold_id >= 0 && connectBodyParts[b].target_hold_id < holds_pos.Length)
            {
                Vector3 localPosRelToHips = hips.InverseTransformPoint(holds_pos[connectBodyParts[b].target_hold_id].position);
                if (float.IsNaN(localPosRelToHips.magnitude) || float.IsInfinity(localPosRelToHips.magnitude))
                {
                    localPosRelToHips = holds_pos[connectBodyParts[b].target_hold_id].position - hips.position;
                }
                AddVectorObs(localPosRelToHips);
            }
            else
            {
                Vector3 localPosRelToHips = new Vector3(-1f, -1f, -1f);
                AddVectorObs(localPosRelToHips);
            }
        }
        // 247 right until spline init vals
        if (useSpline)
        {
            for (int i = 0; i < nActionSize; i++)
            {
                if (float.IsNaN(spline[i].currentValue) || float.IsNaN(spline[i].currentDerivativeValue))
                {
                    spline[i].setState(0f, 0f);
                }
                AddVectorObs(spline[i].currentValue);
                AddVectorObs(spline[i].currentDerivativeValue);
            }
        }
        return;
    }
    
    public override void AgentAction(float[] vectorAction, string textAction)
    {
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
                        sampleVectorAction[i] = vectorAction[i];
                        _sampledActionSpline[i] = vectorAction[i];
                    }
                    
                    if (currentDecisionStep > 0)
                    {
                        num_spline_done++;
                    }
                    currentDecisionStep = 0;
                    counter_spline_not_updated = 0;
                    spline_steps_needed = (minSplineStep + (int)(((sampleVectorAction[0] + 1f) / 2f) * (maxSplineStep - minSplineStep))) * 5;
                }
            }
            else
            {
                if (isNewDecisionStep)
                {
                    int s0 = (minSplineStep + (int)(((sampleVectorAction[0] + 1f) / 2f) * (maxSplineStep - minSplineStep))) * 5 - (currentDecisionStep - 1);
                    vectorAction[0] = (((s0 / 5f) - minSplineStep) / (float)(maxSplineStep - minSplineStep)) * 2f - 1f;
                    _sampledActionSpline[0] = vectorAction[0];
                    for (int i = 1; i < vectorAction.Length; i++)
                    {
                        vectorAction[i] = sampleVectorAction[i];
                        _sampledActionSpline[i] = sampleVectorAction[i];
                    }

                    //for (int i = 0; i < vectorAction.Length; i++)
                    //{
                    //    if (float.IsNaN(vectorAction[i]))
                    //    {
                    //        vectorAction[i] = 0.0f;
                    //    }
                    //    vectorAction[i] = Mathf.Clamp(vectorAction[i], -1f, 1f);
                    //    sampleVectorAction[i] = (vectorAction[i] + sampleVectorAction[i]) / 2.0f;
                    //    _sampledActionSpline[i] = sampleVectorAction[i];
                    //}
                    //spline_steps_needed = (minSplineStep + (int)(((sampleVectorAction[0] + 1f) / 2f) * (maxSplineStep - minSplineStep))) * 5;

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
                    if (reset_each_transition)
                        MyDoneFunc(false, false);

                    succeeded_transition = true;

                    // during training, we do not want to focuse more on one path
                    if (isTrajectoryForked)
                    {
                        if (task_planned_num > ClimberAcademy.max_task_in_episode)
                            MyDoneFunc(false, true);
                    }
                    else
                    {
                        // this part is for reporting as well as for 25% training samples
                        if (task_planned_num > 0)
                            MyDoneFunc(false, true);
                    }
                }
                if (!reset_each_transition)
                    SaveInTrajectory();
            }
            else
            {
                if (isTargetSet && (counter_agent_running - task_counter_step > 500 || num_spline_done >= max_spline_used))// 
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

                    if (isTrajectoryForked)
                    {
                        if (task_failed_num > ClimberAcademy.max_num_fails_in_episode)
                        {
                            MyDoneFunc(false, true);
                        }
                        if (task_planned_num > ClimberAcademy.max_task_in_episode)
                            MyDoneFunc(false, true);
                    }
                    else
                    {
                        // this part is for reporting as well as for 25% training samples
                        if (task_failed_num > 0)
                        {
                            MyDoneFunc(false, true);
                        }
                        // this part is for reporting as well as for 25% training samples
                        if (task_planned_num > 0)
                            MyDoneFunc(false, true);
                    }
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
                    int nValuesPerSegment = 1 + nActionSize;
                    
                    _sampledActionSpline[0] = (minSplineStep + (int)(((_sampledActionSpline[0] + 1f) / 2f) * (maxSplineStep - minSplineStep))) * 0.05f;
                    _sampledActionSpline[nValuesPerSegment] = (minSplineStep + (int)(((_sampledActionSpline[nValuesPerSegment] + 1f) / 2f) * (maxSplineStep - minSplineStep))) * 0.05f;

                    if (agentParameters.onDemandDecision)
                        agentParameters.numberOfActionsBetweenDecisions = spline_steps_needed;// (int)(_sampledActionSpline[0] / Time.fixedDeltaTime);
                }
            }

            if (useSpline)
            {
                GenerateControlValues(counter_spline_not_updated);//currentDecisionStep
                if (currentDecisionStep % 5 == 0)
                {
                    ApplyControl(interpolatedControl);
                }
            }

            ApplyConnectDisconnect();

            // Add relative reward to the current dis to the target
            if (isTargetSet)
            {
                float k_strenght = 0.02f * ((HighLevelGuide.loggedStep) / 1e6f);
                if (k_strenght > 0.2f)
                    k_strenght = 0.2f;
                float k_target = 1.0f - k_strenght;
                for (int b = 0; b < 4; b++)
                {
                    if (connectBodyParts[b].target_hold_id >= 0 && !connectBodyParts[b].IsConnected)
                    {
                        float c_dis = (holds_pos[connectBodyParts[b].target_hold_id].localPosition
                            - connectBodyParts[b]._bodyPartInfo.transform.localPosition).magnitude;

                        if (trajectoryNum == 0)
                        {
                            Vector3 _dir = connectBodyParts[b]._bodyPartInfo.transform.forward;
                            if (b == 3)
                                _dir = connectBodyParts[b]._bodyPartInfo.transform.right;
                            else if (b == 2)
                                _dir = -connectBodyParts[b]._bodyPartInfo.transform.right;
                            Debug.DrawLine(connectBodyParts[b]._bodyPartInfo.transform.position, holds_pos[connectBodyParts[b].target_hold_id].position);
                            Debug.DrawLine(connectBodyParts[b]._bodyPartInfo.transform.position, connectBodyParts[b]._bodyPartInfo.transform.position + connectBodyParts[b].GetRigidBody().velocity.normalized, Color.red);
                            Debug.DrawLine(connectBodyParts[b]._bodyPartInfo.transform.position, connectBodyParts[b]._bodyPartInfo.transform.position + _dir, Color.blue);
                        }

                        if (useSpline)
                        {
                            AddReward(Mathf.Exp(-4.0f * c_dis) / (float)(5f + 1e-6f));
                        }
                        else
                        {
                            float targetDisReward = Mathf.Exp(-4.0f * c_dis) / (float)(agentParameters.numberOfActionsBetweenDecisions + 1e-6);
                            AddReward(k_target * targetDisReward);
                        }
                    }
                }

                float dis_Strength = GetDisStrength(vectorAction);
                float strengthReward = Mathf.Exp(-4.0f * dis_Strength) / (float)(agentParameters.numberOfActionsBetweenDecisions + 1e-6);
                AddReward(k_strenght * strengthReward);
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
                if (counter_agent_running - task_counter_step > 500)// || num_spline_done > max_spline_used
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
                    int nValuesPerSegment = 1 + nActionSize;
                    for (int i = 0; i < vectorAction.Length; i++)
                    {
                        if (float.IsNaN(vectorAction[i]))
                        {
                            vectorAction[i] = 0.0f;
                        }
                        _sampledActionSpline[i] = Mathf.Clamp(vectorAction[i], -1f, 1f);
                    }
                    _sampledActionSpline[0] = (minSplineStep + (int)(((_sampledActionSpline[0] + 1f) / 2f) * (maxSplineStep - minSplineStep))) * 0.05f;
                    _sampledActionSpline[nValuesPerSegment] = (minSplineStep + (int)(((_sampledActionSpline[nValuesPerSegment] + 1f) / 2f) * (maxSplineStep - minSplineStep))) * 0.05f;

                    if (agentParameters.onDemandDecision)
                        agentParameters.numberOfActionsBetweenDecisions = spline_steps_needed;// (int)(_sampledActionSpline[0] / Time.fixedDeltaTime);
                }
            }

            if (useSpline)
            {
                GenerateControlValues(counter_spline_not_updated);
                if (currentDecisionStep % 5 == 0)
                {
                    ApplyControl(interpolatedControl);
                }
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
            if (b <= 1)
            {
                if (connectBodyParts[2].IsConnected || connectBodyParts[3].IsConnected)
                {
                    connectBodyParts[b].connectBodyPart();
                }
                else
                {
                    connectBodyParts[b].disconnectBodyPart();
                }
            }
            else
            {
                connectBodyParts[b].connectBodyPart();
            }
        }
        return;
    }

    /// <summary>
    /// Only change the joint settings based on decision frequency.
    /// </summary>
    void IncrementDecisionTimer()
    {
        counter_agent_running++;

        if (counter_agent_running % 5 == 0)
        {
            if (!flag_test_performance)
            {
                if (!isTrajectoryForked)
                {
                    Vector3 com = GetCOM();
                    HighLevelGuide.AddSitePos(com.x, com.y, GetReward(), succeeded_transition);
                }
            }
            SaveInTrajectory();
        }

        if (!useSpline)
        {
            if (currentDecisionStep == agentParameters.numberOfActionsBetweenDecisions ||
                agentParameters.numberOfActionsBetweenDecisions == 1)
            {
                currentDecisionStep = 1;
                isNewDecisionStep = true;
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

            if (currentDecisionStep % 5 == 0)
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
        float max_r = 0.5f;
        
        // randomize scene
        for (int h = 0; h < holds_pos.Length; h++)
        {
            if (flag_test_performance && ClimberAcademy.Tools.IsSetAContainsHoldID(init_hold_ids, h))
                continue;

            holds_pos[h].transform.localPosition = HighLevelGuide.GetInitHoldPosition(h)
                                                    + Random.Range(0.1f, max_r) * (Quaternion.Euler(0, 0, Random.Range(0, 360)) * Vector3.right);
            
            if (holds_pos[h].transform.localPosition.x < HighLevelGuide.limitedXArea[0])
                holds_pos[h].transform.localPosition.Set(HighLevelGuide.limitedXArea[0], holds_pos[h].transform.localPosition.y, holds_pos[h].transform.localPosition.z);
            if (holds_pos[h].transform.localPosition.x > HighLevelGuide.limitedXArea[1])
                holds_pos[h].transform.localPosition.Set(HighLevelGuide.limitedXArea[1], holds_pos[h].transform.localPosition.y, holds_pos[h].transform.localPosition.z);

            if (holds_pos[h].transform.localPosition.y < HighLevelGuide.limitedYArea[0])
                holds_pos[h].transform.localPosition.Set(holds_pos[h].transform.localPosition.x, HighLevelGuide.limitedYArea[0], holds_pos[h].transform.localPosition.z);
            if (holds_pos[h].transform.localPosition.y > HighLevelGuide.limitedYArea[1])
                holds_pos[h].transform.localPosition.Set(holds_pos[h].transform.localPosition.x, HighLevelGuide.limitedYArea[1], holds_pos[h].transform.localPosition.z);
        }

        if (flag_test_performance)
        {
            if (agent_state_ids.Count > 0)
            {
                // override the target selection to test only one limb!
                for (int b = 0; b < 4; b++)
                {
                    target_hold_ids[b] = init_hold_ids[b];
                    connectBodyParts[b].target_hold_id = init_hold_ids[b];
                }
                target_hold_ids[0] = 0;
                connectBodyParts[0].target_hold_id = 0;
            }

            max_r = 3.0f;
            // randomize scene
            for (int b = 0; b < 4; b++)
            {
                if (init_hold_ids[b] != target_hold_ids[b] && target_hold_ids[b] > -1)
                {
                    int h = target_hold_ids[b];
                    if (ClimberAcademy.Tools.IsSetAContainsHoldID(init_hold_ids, h))
                    {
                        h = 0;
                        target_hold_ids[b] = h;
                        connectBodyParts[b].target_hold_id = h;
                    }
                    holds_pos[h].transform.localPosition = HighLevelGuide.GetInitHoldPosition(init_hold_ids[b])
                                                        + Random.Range(0.5f, max_r) * (Quaternion.Euler(0, 0, Random.Range(0, 360)) * Vector3.right);

                    if (holds_pos[h].transform.localPosition.x < HighLevelGuide.limitedXArea[0])
                        holds_pos[h].transform.localPosition.Set(HighLevelGuide.limitedXArea[0], holds_pos[h].transform.localPosition.y, holds_pos[h].transform.localPosition.z);
                    if (holds_pos[h].transform.localPosition.x > HighLevelGuide.limitedXArea[1])
                        holds_pos[h].transform.localPosition.Set(HighLevelGuide.limitedXArea[1], holds_pos[h].transform.localPosition.y, holds_pos[h].transform.localPosition.z);

                    if (holds_pos[h].transform.localPosition.y < HighLevelGuide.limitedYArea[0])
                        holds_pos[h].transform.localPosition.Set(holds_pos[h].transform.localPosition.x, HighLevelGuide.limitedYArea[0], holds_pos[h].transform.localPosition.z);
                    if (holds_pos[h].transform.localPosition.y > HighLevelGuide.limitedYArea[1])
                        holds_pos[h].transform.localPosition.Set(holds_pos[h].transform.localPosition.x, HighLevelGuide.limitedYArea[1], holds_pos[h].transform.localPosition.z);
                }
                
            }

            targetPlannedHoldPos = GetTargetedHoldPos();
        }

        return;
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    int current_policy_update = 0;
    public override void AgentReset()
    {
        if (!useAgentInterface)
        {
            for (int b = 0; b < 4; b++)
            {
                planned_target_ids[b] = -1;
            }

            if (flag_test_performance)
            {
                if ((int)(connectBodyParts[2].cHoldId / 4) == 2 || (int)(connectBodyParts[3].cHoldId / 4) == 2)
                {
                    if (targetPlannedHoldPos.magnitude > 0)
                    {
                        HighLevelGuide.AddSitePos(targetPlannedHoldPos.x, targetPlannedHoldPos.y, GetCumulativeReward(), succeeded_transition);
                        HighLevelGuide.AddTransitionReward(GetCumulativeReward(), counter_agent_running / 5, num_spline_done, isTrajectoryForked);
                    }
                }
                HighLevelGuide.GetCurrentAvgAccumulativeReward();

                isTrajectoryForked = false;

                if (agent_state_ids.Count > 0)
                {
                    // chosen init state to load for random training
                    LoadState(agent_state_ids[0]);
                }
            }
            else
            {
                HighLevelGuide.AddTransitionReward(GetCumulativeReward(), counter_agent_running / 5, num_spline_done, isTrajectoryForked);
            }

            SaveInTrajectory();

            if (fatherNodeID >= 0)
            {
                bool flag_update = false;
                for (int b = 0; b < 4 && !flag_update; b++)
                {
                    if (HighLevelGuide.graphNodes[fatherNodeID].holdIds[b] != target_hold_ids[b] && target_hold_ids[b] > -1)
                    {
                        flag_update = true;
                    }
                }
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
                Vector2Int _stateID_forkedStatus = HighLevelGuide.GetRandomState(trajectoryNum, flag_test_performance, ref planned_target_ids);
                
                LoadState(_stateID_forkedStatus[0]);
                
                isTrajectoryForked = _stateID_forkedStatus[1] == 1;

                if (flag_test_performance)
                {
                    if (agent_state_ids.Count > 0)
                    {
                        // chosen init state to load for random training
                        LoadState(agent_state_ids[0]);
                        isTrajectoryForked = false;
                    }
                    else
                    {
                        isTrajectoryForked = true;
                    }
                }
                
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

            current_count_limb_movment = Random.Range((int)ClimberAcademy.min_count_limb_movement - 1, (int)ClimberAcademy.max_count_limb_movement);

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
            if (!agent_tree_task.Contains(ClimberAcademy.StanceToKey(HighLevelGuide.graphNodes[fatherNodeID].holdIds)))
                agent_tree_task.Add(ClimberAcademy.StanceToKey(HighLevelGuide.graphNodes[fatherNodeID].holdIds));

            SetNextTargetHoldIds(true);

            current_policy_update = HighLevelGuide.numUpdatedPolicy;

            if (flag_test_performance)
            {
                if (agent_state_ids.Count > 0)
                {
                    RandomizeScene();
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
        if (flag_test_performance)
        {
            if (ClimberAcademy.Tools.GetNumConnectedLimbs(cHoldIDs) < minConnectedLimbsInitHolds)
            {
                return;
            }

            if (agent_state_ids.Count > 0)
            {
                return;
            }
        }

        current_step++;
        while (current_step >= agent_state_ids.Count)
        {
            agent_state_ids.Add(-1);
        }

        if (agent_state_ids[current_step] == -1)
        {
            agent_state_ids[current_step] = HighLevelGuide.GetNextFreeState(jdController.bodyPartsList.Count, spline.Count);
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
        ClimberAcademy.HumanoidBodyState _cState = HighLevelGuide.savedStates[_stateID];

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
            if (connectBodyParts[b].cHoldId >= 0)
                _cState.current_hold_pos[b] = holds_pos[connectBodyParts[b].cHoldId].localPosition;

            _cState.connectorTouchingGround[b] = connectBodyParts[b].groundContact.touchingGround;
            _cState.connectorTouchingWall[b] = connectBodyParts[b].wallContact.touchingWall;
            _cState.connectorTouchingHold[b] = connectBodyParts[b].holdContact.touchingHold;

            _cState.connectorPos[b] = connectBodyParts[b].GetRigidBody().transform.localPosition;
            _cState.connectorRot[b] = connectBodyParts[b].GetRigidBody().transform.localRotation;
            _cState.connectorVel[b] = connectBodyParts[b].GetRigidBody().velocity;
            _cState.connectorAVel[b] = connectBodyParts[b].GetRigidBody().angularVelocity;
        }

        for (int i = 0; i < jdController.bodyPartsList.Count; i++)
        {
            _cState.pos[i] = jdController.bodyPartsList[i].rb.transform.localPosition;
            _cState.rot[i] = jdController.bodyPartsList[i].rb.transform.localRotation;
            _cState.vel[i] = jdController.bodyPartsList[i].rb.velocity;
            _cState.aVel[i] = jdController.bodyPartsList[i].rb.angularVelocity;

            _cState.touchingGround[i] = jdController.bodyPartsList[i].groundContact.touchingGround;
            _cState.touchingWall[i] = jdController.bodyPartsList[i].wallContact.touchingWall;

            _cState.touchingHold[i] = jdController.bodyPartsList[i].holdContact.touchingHold;
            _cState.touchingHoldId[i] = jdController.bodyPartsList[i].holdContact.hold_id;

            _cState.touchingTarget[i] = jdController.bodyPartsList[i].targetContact.touchingTarget;
            
        }
        return;
    }

    public void LoadState(int _stateID)
    {
        ClimberAcademy.HumanoidBodyState _cState = HighLevelGuide.savedStates[_stateID];

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
                if (!flag_test_performance)
                {
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
                }
                else
                {
                    int hand_row = Mathf.Max(row_cur_ids[2], row_cur_ids[3]);
                    row_displacement = 2 - hand_row;

                    int hand_col = col_cur_ids[3];
                    col_displacement = 2 - hand_col;
                    if (hand_col < 0)
                    {
                        col_displacement = 1 - col_cur_ids[2];
                    }
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
        // loading the position of saved holds for the current state of the climber
        for (int b = 0; b < 4; b++)
        {
            if (current_hold_ids[b] >= 0)
                holds_pos[current_hold_ids[b]].localPosition = current_hold_pos[b];

            connectBodyParts[b].target_hold_id = -1;
            connectBodyParts[b].disconnectBodyPart();

            connectBodyParts[b].groundContact.touchingGround = _cState.connectorTouchingGround[b];
            connectBodyParts[b].wallContact.touchingWall = _cState.connectorTouchingWall[b];
            if (current_hold_ids[b] < 0)
                connectBodyParts[b].holdContact.touchingHold = _cState.connectorTouchingHold[b];
            
        }

        for (int i = 0; i < jdController.bodyPartsList.Count; i++)
        {
            jdController.bodyPartsList[i].rb.transform.localPosition = _cState.pos[i] + hip_displacement;
            jdController.bodyPartsList[i].rb.transform.localRotation = _cState.rot[i];
            jdController.bodyPartsList[i].rb.velocity = _cState.vel[i];
            jdController.bodyPartsList[i].rb.angularVelocity = _cState.aVel[i];

            jdController.bodyPartsList[i].groundContact.touchingGround = _cState.touchingGround[i];
            jdController.bodyPartsList[i].wallContact.touchingWall = _cState.touchingWall[i];

            jdController.bodyPartsList[i].holdContact.touchingHold = _cState.touchingHold[i];
            jdController.bodyPartsList[i].holdContact.hold_id = _cState.touchingHoldId[i];

            jdController.bodyPartsList[i].targetContact.touchingTarget = _cState.touchingTarget[i];
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
        Vector3 com = new Vector3();
        float total_mass = 0;
        for (int i = 0; i < jdController.bodyPartsList.Count; i++)
        {
            Vector3 p = jdController.bodyPartsList[i].rb.transform.localPosition;
            float m = jdController.bodyPartsList[i].rb.mass;
            total_mass += m;
            com += (p * m);
        }
        return com / total_mass;
    }

    Vector3 GetTargetedHoldPos()
    {
        Vector3 tPos = new Vector3(0f, 0f, 0f);
        int c = 0;
        for (int b = 0; b < 4; b++)
        {
            if (connectBodyParts[b].target_hold_id != init_hold_ids[b])
            {
                if (connectBodyParts[b].target_hold_id >= 0 && connectBodyParts[b].target_hold_id < holds_pos.Length)
                {
                    tPos += holds_pos[connectBodyParts[b].target_hold_id].localPosition;
                    c++;
                }
            }
        }
        
        return tPos / (c + 1e-6f);
    }
}
