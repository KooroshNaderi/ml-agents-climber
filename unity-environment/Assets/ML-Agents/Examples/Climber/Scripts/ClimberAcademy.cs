using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using System.IO;

public class ClimberAcademy : Academy
{
    const int MaxHoldCount = 16;

    public class HumanoidBodyState
    {
        public HumanoidBodyState(int boneCount, int splineInitCount)
        {
            pos = new Vector3[boneCount];
            rot = new Quaternion[boneCount];
            vel = new Vector3[boneCount];
            aVel = new Vector3[boneCount];

            connectorPos = new Vector3[4];
            connectorRot = new Quaternion[4];
            connectorVel = new Vector3[4];
            connectorAVel = new Vector3[4];

            touchingGround = new bool[boneCount];
            touchingWall = new bool[boneCount];
            touchingHold = new bool[boneCount];
            touchingHoldId = new int[boneCount];
            touchingTarget = new bool[boneCount];

            spline_init_values = new Vector2[splineInitCount];
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

        // defines whether hands/feet are connected to a hold given the hold id (-1 means the body is not connected)
        public int[] current_hold_ids = { -1, -1, -1, -1 };
        // defines relative positions of the connected holds to current object world
        public Vector3[] current_hold_pos = { new Vector3(), new Vector3(), new Vector3(), new Vector3() };

        public bool[] connectorTouchingGround = { false, false, false, false };
        public bool[] connectorTouchingWall = { false, false, false, false };
        public bool[] connectorTouchingHold = { false, false, false, false };

        public Vector3[] connectorPos;
        public Quaternion[] connectorRot;
        public Vector3[] connectorVel;
        public Vector3[] connectorAVel;

        public Vector2[] spline_init_values;

        public float current_accumulative_reward = 0f;
        public int current_step = 0;
    };

    public class ClimberStance
    {
        public int agent_state_id = -1;
        public float stance_value = 0.0f;
        public int stance_id = -1;
    };

    public class ClimberStanceTarget
    {
        public List<int> agent_state_ids = new List<int>();
        public int valid_state_counter = 0;
        public float value = 0.0f;
        public int stance_id = -1;
        public int[] targetHoldIds = { -1, -1, -1, -1 };
    };

    public class ContextManager
    {
        private const float holdSize = 0.25f;
        List<Vector3> _holdsPositions = new List<Vector3>();// not-biased positions

        Vector3 goal_pos = new Vector3(0, 0, 0);
        //public int goal_hold_id = -1;

        public ContextManager(GameObject _instantObj)
        {
            for (int _c = 0; _c < _instantObj.transform.childCount; _c++)
            {
                if (_instantObj.transform.GetChild(_c).name.Contains("h"))
                {
                    Vector3 pos = _instantObj.transform.GetChild(_c).position - _instantObj.transform.position;
                    _holdsPositions.Add(pos);
                }
            }

            for (int i = _holdsPositions.Count - 1; i >= _holdsPositions.Count - 4; i--)
            {
                goal_pos += _holdsPositions[i] / 4.0f;
            }
        }

        public static float HoldSize
        {
            get { return holdSize; }
        }

        public Vector3 HoldPosition(int i)
        {
            if (i < 0 || i >= _holdsPositions.Count)
                return new Vector3(0, 0, 0);
            return _holdsPositions[i];
        }

        public int NumHolds()
        {
            return _holdsPositions.Count;
        }

        void GetHoldsInRadius(Vector3 dP, float r, ref List<int> ret_holds_ids)
        {
            for (int i = 0; i < _holdsPositions.Count; i++)
            {
                Vector3 hold_i = HoldPosition(i);

                float cDis = (hold_i - dP).magnitude;
                if (cDis < r)
                {
                    ret_holds_ids.Add(i);
                }
            }
        }

        public float GetClimberRadius()
        {
            return 2.0f;
        }

        public Vector3 GetHoldStancePosFrom(int[] _hold_ids, ref List<Vector3> _hold_points, ref float mSize)
        {
            Vector3 midPoint = new Vector3(0.0f, 0.0f, 0.0f);
            mSize = 0.0f;
            _hold_points.Clear();
            for (int i = 0; i < _hold_ids.Length; i++)
            {
                if (_hold_ids[i] != -1)
                {
                    _hold_points.Add(HoldPosition(_hold_ids[i]));
                    midPoint += _hold_points[i];
                    mSize++;
                }
                else
                    _hold_points.Add(new Vector3(0, 0, 0));
            }

            if (mSize > 0)
            {
                midPoint = midPoint / mSize;
            }
            else//we are on the ground
            {
                midPoint = new Vector3(0.0f, 1.1f, 0.85f);
            }
            return midPoint;
        }

        public Vector3 GetExpectedLimbPosition(int _hold_id)
        {
            if (_hold_id != -1)
            {
                return HoldPosition(_hold_id);
            }
            
            return new Vector3(0f, -2f, 0f);
        }

        public Vector3 GetExpectedMidHoldStancePos(int[] _hold_ids)
        {
            Vector3 midPoint = new Vector3(0.0f, 0.0f, 0.0f);
            for (int i = 0; i < _hold_ids.Length; i++)
            {
                midPoint += GetExpectedLimbPosition(_hold_ids[i]) / 4.0f;
            }
            
            return midPoint;
        }

        public Vector3 GetMidHoldStancePos(int[] _hold_ids)
        {
            Vector3 midPoint = new Vector3(0.0f, 0.0f, 0.0f);
            float mSize = 0.0f;
            for (int i = 0; i < _hold_ids.Length; i++)
            {
                if (_hold_ids[i] != -1)
                {
                    Vector3 p = HoldPosition(_hold_ids[i]);
                    midPoint += p;
                    mSize++;
                }
            }

            if (mSize > 0)
            {
                midPoint = midPoint / mSize;
            }
            else//we are on the ground
            {
                midPoint = new Vector3(0.0f, 1.1f, 0.85f);
            }
            return midPoint;
        }

        public Vector3 GetGoalHold()
        {
            return goal_pos;
        }
    };

    public class ActionMovement : System.IEquatable<ActionMovement>
    {
        public List<Vector2Int> action_limbs = new List<Vector2Int>();
        public int ActionMovementId { get; set; }

        public ActionMovement()
        {
            // at most two limbs movement
            action_limbs.Add(new Vector2Int(-1, -1));
            action_limbs.Add(new Vector2Int(-1, -1));
            CalculateHashID();
        }

        void CalculateHashID()
        {
            ActionMovementId = 0;
            if (action_limbs.Count >= 1)
            {
                ActionMovementId = ActionMovementId + (action_limbs[0][0] + 1);
                ActionMovementId = ActionMovementId << 8;
                ActionMovementId = ActionMovementId + (action_limbs[0][1] + 1);
                ActionMovementId = ActionMovementId << 8;
            }
            if (action_limbs.Count >= 2)
            {
                ActionMovementId = ActionMovementId + (action_limbs[1][0] + 1);
                ActionMovementId = ActionMovementId << 8;
                ActionMovementId = ActionMovementId + (action_limbs[1][1] + 1);
            }
        }

        public void AddAction(Vector2Int a)
        {
            // alwayse remove first and choose the right position for the new action
            action_limbs.RemoveAt(0);
            if (action_limbs[0][0] <= a[0])
            {
                action_limbs.Add(a);
            }
            else
            {
                action_limbs.Insert(action_limbs.Count - 1, a);
            }
            CalculateHashID();
        }

        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            ActionMovement objAsPart = obj as ActionMovement;
            if (objAsPart == null) return false;
            else return Equals(objAsPart);
        }

        public override int GetHashCode()
        {
            return ActionMovementId;
        }

        public bool Equals(ActionMovement other)
        {
            if (other == null) return false;
            return (this.ActionMovementId.Equals(other.ActionMovementId));
        }

    }

    public class GraphNode
    {
        const int index_1_limb = 0;
        const int index_2_limb = 1;

        public int updated_node_num = 0;
        public float preNodeRLValue = 0;
        public float NodeRLValue = 0;
        public int NumTriedTransitions = 0;

        public int nodeIdx = -1;
        public int[] holdIds = { -1, -1, -1, -1 };
        public float costToNode = 0;

        public int bestParentNodeIdx = -1;
        
        public List<int> humanoidStateIds = new List<int>();
        public Vector3 hipPos = new Vector3(0, 0, 0);

        // 0:1-limb movement, 1:2-limbs movements
        public List<ActionMovement>[] possible_actions = { new List<ActionMovement>(), new List<ActionMovement>() };
        public List<float>[] successRate = { new List<float>(), new List<float>() };
        public List<int>[] counterSeenAction = { new List<int>(), new List<int>() };
        public List<float>[] preSuccessRate = { new List<float>(), new List<float>() };
        public List<int>[] counter_fixed_discovery = { new List<int>(), new List<int>() };
        public List<int>[] childNodeIds = { new List<int>(), new List<int>() };

        public bool isNodeOnLadderClimbing = false;
        public int planning_index_on_ladder = -1;

        // each node has some holds around them
        List<int> rh_hold_ids = new List<int>();
        List<int> lh_hold_ids = new List<int>();
        List<int> rl_hold_ids = new List<int>();
        List<int> ll_hold_ids = new List<int>();

        int[] added_1limb_transitions = { 0, 0, 0, 0 };
        Vector2Int[] added_2limb_transtions = { new Vector2Int(0, 0), // rh-rl
                                                new Vector2Int(0, 0), // rh-ll
                                                new Vector2Int(0, 0), // lh-rl
                                                new Vector2Int(0, 0), // lh-ll
                                                new Vector2Int(0, 0)  // rl-ll 
                                              };

        public bool isFixedNode = false;
        
        int GetListCountGivenLimbID(int limb_id)
        {
            switch (limb_id)
            {
                case 0:
                    return ll_hold_ids.Count;
                case 1:
                    return rl_hold_ids.Count;
                case 2:
                    return lh_hold_ids.Count;
                case 3:
                    return rh_hold_ids.Count;
            }
            return -1;
        }

        public float GetNodeValue()
        {
            //float sum_rate = 0.0f;
            //for (int plan_count_limb_movement = 0; plan_count_limb_movement < possible_actions.Length; plan_count_limb_movement++)
            //{
            //    for (int c = 0; c < possible_actions[plan_count_limb_movement].Count; c++)
            //    {
            //        float c_rate = GetSuccessRate(c, plan_count_limb_movement, true);
            //        if (c_rate > ClimberAcademy.minSuccessRate)
            //            sum_rate += c_rate;
            //    }
            //    sum_rate += 1.0f * childNodeIds[plan_count_limb_movement].Count;
            //}
            //return sum_rate;

            float nValSuccess = (NodeRLValue / (NumTriedTransitions + 1e-6f));

            return preNodeRLValue + 0.1f * (nValSuccess - preNodeRLValue);
        }
        
        public Vector2Int GetRandomActionOrChildIndex(int plan_count_limb_movement, bool isTrajectoryForked)
        {
            // calculate probabilites on the stance graph
            float sum_rate = 0.0f;
            for (int c = 0; c < possible_actions[plan_count_limb_movement].Count; c++)
            {
                float c_rate = GetSuccessRate(c, plan_count_limb_movement, isTrajectoryForked);
                if (c_rate > ClimberAcademy.minSuccessRate)
                    sum_rate += c_rate;
            }
            sum_rate += 1.0f * childNodeIds[plan_count_limb_movement].Count;

            float p = Random.Range(0.0f, 1.0f);
            int chosen_action_index = -1;
            float s_value = 0.0f;
            for (int c = 0; c < possible_actions[plan_count_limb_movement].Count && chosen_action_index == -1; c++)
            {
                float c_rate = GetSuccessRate(c, plan_count_limb_movement, isTrajectoryForked);
                if (c_rate > ClimberAcademy.minSuccessRate)
                {
                    s_value += c_rate;
                    if ((s_value / sum_rate) >= p)
                    {
                        chosen_action_index = c;
                    }
                }
            }

            if (chosen_action_index > -1 && chosen_action_index < possible_actions[plan_count_limb_movement].Count)
            {
                return new Vector2Int(-1, chosen_action_index);
            }
            else
            {
                for (int c = 0; c < childNodeIds[plan_count_limb_movement].Count && chosen_action_index == -1; c++)
                {
                    float c_rate = 1.0f;
                    s_value += c_rate;
                    if ((s_value / sum_rate) >= p)
                    {
                        chosen_action_index = c;
                    }
                }
                if (chosen_action_index > -1 && chosen_action_index < childNodeIds[plan_count_limb_movement].Count)
                {
                    int chosen_child_index = childNodeIds[plan_count_limb_movement][chosen_action_index];
                    
                    return new Vector2Int(chosen_child_index, -1);
                }
            }
            
            return new Vector2Int(-1, -1);
        }

        public int[] GetTargetHoldIDs(int plan_count_limb_movement, int action_index)
        {
            int[] nHoldIDs = { holdIds[0], holdIds[1], holdIds[2], holdIds[3] };

            for (int i = 0; i < 2; i++)
            {
                if (possible_actions[plan_count_limb_movement][action_index].action_limbs[i][0] > -1)
                {
                    nHoldIDs[possible_actions[plan_count_limb_movement][action_index].action_limbs[i][0]]
                        = possible_actions[plan_count_limb_movement][action_index].action_limbs[i][1];
                }
            }

            return nHoldIDs;
        }

        int GetPossibleHoldIDFromLimb(int limb_id, int tra_id)
        {
            int nHold = -1;
            switch (limb_id)
            {
                case 0:
                    nHold = ll_hold_ids[tra_id];
                    break;
                case 1:
                    nHold = rl_hold_ids[tra_id];
                    break;
                case 2:
                    nHold = lh_hold_ids[tra_id];
                    break;
                case 3:
                    nHold = rh_hold_ids[tra_id];
                    break;
            }
            return nHold;
        }

        public void CompleteActionList()
        {
            for (int limb_id = 0; limb_id < added_1limb_transitions.Length; limb_id++)
            {
                if (added_1limb_transitions[limb_id] < GetListCountGivenLimbID(limb_id))
                {
                    int nHold = GetPossibleHoldIDFromLimb(limb_id, added_1limb_transitions[limb_id]);
                    added_1limb_transitions[limb_id]++;
                    AddAction(limb_id, nHold);
                }
            }
            if (ClimberAcademy.max_count_limb_movement == 2)
            {
                for (int tra_id = 0; tra_id < added_2limb_transtions.Length; tra_id++)
                {
                    int limb_id_1 = -1;
                    int limb_id_2 = -1;
                    switch (tra_id)
                    {
                        case 0:
                            limb_id_1 = 3;
                            limb_id_2 = 1;
                            break;
                        case 1:
                            limb_id_1 = 3;
                            limb_id_2 = 0;
                            break;
                        case 2:
                            limb_id_1 = 2;
                            limb_id_2 = 1;
                            break;
                        case 3:
                            limb_id_1 = 2;
                            limb_id_2 = 0;
                            break;
                        case 4:
                            limb_id_1 = 1;
                            limb_id_2 = 0;
                            break;
                    }
                    if (added_2limb_transtions[tra_id][0] < GetListCountGivenLimbID(limb_id_1))
                    {
                        if (added_2limb_transtions[tra_id][1] < GetListCountGivenLimbID(limb_id_2))
                        {
                            int nHold_1 = GetPossibleHoldIDFromLimb(limb_id_1, added_2limb_transtions[tra_id][0]);
                            int nHold_2 = GetPossibleHoldIDFromLimb(limb_id_2, added_2limb_transtions[tra_id][1]);
                            AddAction(limb_id_1, nHold_1, limb_id_2, nHold_2);
                            added_2limb_transtions[tra_id][1]++;
                        }
                        else if (added_2limb_transtions[tra_id][0] < GetListCountGivenLimbID(limb_id_1))
                        {
                            added_2limb_transtions[tra_id][0]++;
                            added_2limb_transtions[tra_id][1] = 0;
                        }
                    }
                }
            }
            return;
        }

        int FindAction(int limb_id, int hold_id)
        {
            Vector2Int n_a = new Vector2Int(limb_id, hold_id);
            ActionMovement newMovement = new ActionMovement();
            newMovement.AddAction(n_a);
            int index = possible_actions[index_1_limb].IndexOf(newMovement);
            return index;
        }

        float GetSuccessRate(int ccIndex, int transition_type, bool isTrajectoryForked)
        {
            // when trajectory is not forked, the trajectory is used for reporting the performance
            // so non-ForkedTrajectories can only choose uniform randomly transitions
            if (!isTrajectoryForked)
            {
                return 1.0f;
            }
            float s_rate = minSuccessRate;
            if (ccIndex != -1)
            {
                s_rate = Mathf.Max(s_rate, successRate[transition_type][ccIndex]);
            }
            return s_rate;
        }

        int AddAction(int limb_id, int hold_id)
        {
            if (limb_id < 0)
                return -1;
            
            int[] nHoldIds = { holdIds[0], holdIds[1], holdIds[2], holdIds[3] };
            nHoldIds[limb_id] = hold_id;

            if (!ClimberAcademy.IsValidTransition(holdIds, nHoldIds))
            {
                return -1;
            }

            Vector2Int n_a = new Vector2Int(limb_id, hold_id);
            ActionMovement newMovement = new ActionMovement();
            newMovement.AddAction(n_a);
            int index = possible_actions[index_1_limb].IndexOf(newMovement);
            if (index > -1)
            {
                return index;
            }
            else
            {
                possible_actions[index_1_limb].Add(newMovement);
                successRate[index_1_limb].Add(0.5f);
                counterSeenAction[index_1_limb].Add(0);
                preSuccessRate[index_1_limb].Add(0);
                counter_fixed_discovery[index_1_limb].Add(0);

                return possible_actions[index_1_limb].Count - 1;
            }
        }

        int AddAction(int limb_id_1, int hold_id_1, int limb_id_2, int hold_id_2)
        {
            if (limb_id_1 < 0 || limb_id_2 < 0)
                return -1;
            
            int[] nHoldIds = { holdIds[0], holdIds[1], holdIds[2], holdIds[3] };
            nHoldIds[limb_id_1] = hold_id_1;
            nHoldIds[limb_id_2] = hold_id_2;

            if (!ClimberAcademy.IsValidTransition(holdIds, nHoldIds))
            {
                return -1;
            }

            ActionMovement newMovement = new ActionMovement();
            newMovement.AddAction(new Vector2Int(limb_id_1, hold_id_1));
            newMovement.AddAction(new Vector2Int(limb_id_2, hold_id_2));
            int index = possible_actions[index_2_limb].IndexOf(newMovement);
            if (index > -1)
            {
                return index;
            }
            else
            {
                possible_actions[index_2_limb].Add(newMovement);
                successRate[index_2_limb].Add(0.5f);
                counterSeenAction[index_2_limb].Add(0);
                preSuccessRate[index_2_limb].Add(0);
                counter_fixed_discovery[index_2_limb].Add(0);

                return possible_actions[index_2_limb].Count - 1;
            }
        }

        public void UpdateHoldIdsForLimb(int limb_id, int hold_id)
        {
            if (limb_id < 0)
                return;

            if (hold_id >= ClimberAcademy.MaxHoldCount)
                hold_id = -1;

            if (hold_id < 0)
                hold_id = -1;

            int[] nHoldIds = { holdIds[0], holdIds[1], holdIds[2], holdIds[3] };
            nHoldIds[limb_id] = hold_id;

            if (!ClimberAcademy.IsValidTransition(holdIds, nHoldIds))
            {
                return;
            }
            
            switch (limb_id)
            {
                case 0:
                    if (!ll_hold_ids.Contains(hold_id))
                        ll_hold_ids.Add(hold_id);
                    break;
                case 1:
                    if (!rl_hold_ids.Contains(hold_id))
                        rl_hold_ids.Add(hold_id);
                    break;
                case 2:
                    if (!lh_hold_ids.Contains(hold_id))
                        lh_hold_ids.Add(hold_id);
                    break;
                case 3:
                    if (!rh_hold_ids.Contains(hold_id))
                        rh_hold_ids.Add(hold_id);
                    break;
            }
            return;
        }
    };

    public class Tools
    {
        public static float GetGussianRandomNoise(float stdDev)
        {
            Random rand = new Random(); //reuse this if you are generating many
            float u1 = 1.0f - Random.Range(0f, 1f); //uniform(0,1] random doubles
            float u2 = 1.0f - Random.Range(0f, 1f);
            float randStdNormal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) *
                         Mathf.Sin(2.0f * Mathf.PI * u2); //random normal(0,1)
            float randNormal = stdDev * randStdNormal; //random normal(mean,stdDev^2)

            return 0.0f;
        }

        public static int GetNumConnectedLimbs(int[] a)
        {
            int c = 0;
            for (int h = 0; h < a.Length; h++)
            {
                if (a[h] > -1)
                {
                    c++;
                }
            }
            return c;
        }

        public static bool IsSetAEqualSetB(int[] a, int[] b)
        {
            if (a.Length != b.Length)
            {
                return false;
            }
            for (int h = 0; h < a.Length; h++)
            {
                if (a[h] != b[h])
                {
                    return false;
                }
            }
            return true;
        }

        public static bool IsSetAContainsHoldID(int[] a, int holdID)
        {
            for (int h = 0; h < a.Length; h++)
            {
                if (a[h] == holdID)
                {
                    return true;
                }
            }
            return false;
        }

        public static int GetDiffBtwSetASetB(int[] set_a, int[] set_b)
        {
            int mCount = 0;
            for (int i = 0; i < set_a.Length; i++)
            {
                if (set_a[i] != set_b[i])
                {
                    mCount++;
                }
            }
            return mCount;
        }
    }

    public GameObject instantObject;
    
    // save humanoid states here for making it accessible to all agents
    public enum TrainingType { RandomTraining = 0, // alwayse starts from init, do random transition until max_task_in_episode is done
                               GraphPSITraining = 1, // use graph success values to store best state in different stances
                               GraphRandomTraining = 2,
                               GraphTreeTraining = 3,
                               LadderClimbing = 4,
                               PSITraining = 5,
                               UniformTraining = 6,
                               PSITargetTraining = 7};

    [HideInInspector] public const TrainingType current_training_type = TrainingType.UniformTraining;
    [HideInInspector] public const int min_count_limb_movement = 1;
    [HideInInspector] public const int max_count_limb_movement = 1;
    [HideInInspector] public const float minSuccessRate = 0.1f;

    [HideInInspector] public const int max_task_in_episode = current_training_type == TrainingType.PSITargetTraining ? 5 : 0;
    [HideInInspector] public const int max_num_fails_in_episode = current_training_type == TrainingType.PSITargetTraining ? 2 : 0;
    [HideInInspector] const string base_file_name = "climber_Spline_";

    // 0.1 is a lot when you have not learnt any movement
    [HideInInspector] float[] minSuccessRateChange = { 0.01f, 0.01f };
    [HideInInspector] public List<HumanoidBodyState> savedStates = new List<HumanoidBodyState>();
    [HideInInspector] public List<GraphNode> graphNodes = new List<GraphNode>();
    [HideInInspector] public Dictionary<ulong, int> graphNodeDict = new Dictionary<ulong, int>();

    private float disToTarget = float.MaxValue;
    private float least_cost_to_goal = float.MaxValue;
    [HideInInspector] public int bestNodeIndex = -1;
    private List<int> path_to_target = new List<int>();
    private ContextManager mContext = null;
    private int _MaxNodeCount = 1;

    [HideInInspector] public List<ClimberAgent> _ControlRigs = new List<ClimberAgent>();
    private List<int> to_fix_graph_nodes = new List<int>();

    ////////////////////////////////////////////////////////// PSI implementation ///////////////////////////////////////////////////////////
    const bool flag_reset_trajectories = true;

    [HideInInspector] const int MaxTrajectoryStoredNumber = 2000;
    
    [HideInInspector] public List<ClimberStance> seenStances = new List<ClimberStance>();
    [HideInInspector] public List<ClimberStanceTarget> seenStanceTargets = new List<ClimberStanceTarget>();

    int seenStanceCounter = 0;
    int seenStanceTargetCounter = 0;

    StreamWriter streamWriter = null;
    int nonForkedEpisodeCounter = 0;
    float nonForked_accumulative_reward = 0.0f;

    int nonForkedTransitionCounter = 0;
    float nonForkedTransition_accumulative_reward = 0.0f;
    int nonForkedTransition_EpisodeLength = 0;
    int nonForkedTransition_NumSpline = 0;

    [HideInInspector] public int loggedStep = 0;
    [HideInInspector] public int numUpdatedPolicy = 0;

    [HideInInspector] public float[] limitedXArea = {-7.0f, 7.0f};
    [HideInInspector] public float[] limitedYArea = { -0.5f, 7.5f };
    const int visited_pos_size_x = 14 * 25;
    const int visited_pos_size_y = 8 * 25;

    float[][] visited_pos_r = new float[visited_pos_size_x][];
    int[][] visited_pos_c = new int[visited_pos_size_x][];
    int[][] visited_pos_s = new int[visited_pos_size_x][];
    int count_to_end = 0;
    public float GetCurrentAvgAccumulativeReward()
    {
        if (nonForkedTransitionCounter > 1000)
        {
            float val = nonForkedTransition_accumulative_reward / nonForkedTransitionCounter;
            
            Debug.Log("Count: " + nonForkedTransitionCounter.ToString() + ", Value: " + (val).ToString());

            nonForkedTransitionCounter = 0;
            nonForkedTransition_accumulative_reward = 0;

            count_to_end++;
            return val;
        }

        if (count_to_end > 10)
        {
#if UNITY_EDITOR
            // Application.Quit() does not work in the editor so
            // UnityEditor.EditorApplication.isPlaying need to be set to false to end the game
            UnityEditor.EditorApplication.isPlaying = false;
#endif
            Application.Quit();

        }
        return 0;
    }

    public void AddSitePos(float x, float y, float r, bool isSucceed)
    {
        int index_x = (int)((x + Mathf.Abs(limitedXArea[0])) * (visited_pos_size_x / 14.0f));
        int index_y = (int)((y + Mathf.Abs(limitedYArea[0])) * (visited_pos_size_y / 8.0f));
        if (index_x < 0) index_x = 0;
        if (index_x >= visited_pos_size_x) index_x = visited_pos_size_x - 1;
        if (index_y < 0) index_y = 0;
        if (index_y >= visited_pos_size_y) index_y = visited_pos_size_y - 1;

        visited_pos_r[index_x][index_y] += r;
        visited_pos_c[index_x][index_y]++;
        if (isSucceed)
        {
            visited_pos_s[index_x][index_y]++;
        }
    }

    public void AddTransitionReward(float accumulative_reward, int episode_length, int num_spline_used, bool isTrajectoryForked)
    {
        if (!isTrajectoryForked)
        {
            nonForkedTransitionCounter++;
            nonForkedTransition_accumulative_reward += accumulative_reward;
            nonForkedTransition_EpisodeLength += episode_length;
            nonForkedTransition_NumSpline += num_spline_used;
        }

        return;
    }

    public void AddPathReward(float accumulative_reward, bool isTrajectoryForked)
    {
        if (!isTrajectoryForked)
        {
            nonForked_accumulative_reward += accumulative_reward;
            nonForkedEpisodeCounter++;
        }
        
        return;
    }

    public void UpdateSeenStance(int current_stance_id)
    {
        if (current_stance_id < 0)
            return;

        float stance_value = graphNodes[current_stance_id].GetNodeValue();
        bool flag_updated = false;
        for (int t = 0; t < seenStanceCounter && !flag_updated; t++)
        {
            if (seenStances[t].stance_id == current_stance_id)
            {
                seenStances[t].stance_value = stance_value;
                flag_updated = true;
            }
        }
        return;
    }

    public void AddStanceTarget(List<int> state_ids, int current_step, int[] target_hold_ids, bool flag_add_stance_target, float val)
    {
        if (current_training_type != TrainingType.PSITargetTraining)
            return;

        if (current_step < 0 || Tools.GetNumConnectedLimbs(target_hold_ids) == 0)
            return;

        int init_stance_id = ReturnNodeId(savedStates[state_ids[0]].current_hold_ids);

        bool flag_exists = false;
        for (int t = 0; t < seenStanceTargetCounter && !flag_exists; t++)
        {
            if (Tools.IsSetAEqualSetB(target_hold_ids, seenStanceTargets[t].targetHoldIds) && init_stance_id == seenStanceTargets[t].stance_id)
            {
                // updating
                seenStanceTargets[t].valid_state_counter = 0;
                for (int id = 0; id < current_step + 1; id++)
                {
                    if (id >= seenStanceTargets[t].agent_state_ids.Count)
                    {
                        seenStanceTargets[t].agent_state_ids.Add(-1);
                    }
                    if (seenStanceTargets[t].agent_state_ids[id] >= 0)
                        CopyState(state_ids[id], seenStanceTargets[t].agent_state_ids[id]);
                    else
                        seenStanceTargets[t].agent_state_ids[id] = CopyState(state_ids[id]);
                    seenStanceTargets[t].valid_state_counter++;
                }
                seenStanceTargets[t].value = val;
                flag_exists = true;
            }
        }

        if (!flag_exists)
        {
            int nIndex = seenStanceTargets.Count - 1;
            if (seenStanceTargetCounter < seenStanceTargets.Count)
            {
                nIndex = seenStanceTargetCounter;
                seenStanceTargetCounter++;
            }
            else
            {
                seenStanceTargets.Add(new ClimberStanceTarget());
                seenStanceTargetCounter = seenStanceTargets.Count;
                nIndex = seenStanceTargets.Count - 1;
            }

            // updating
            seenStanceTargets[nIndex].valid_state_counter = 0;
            for (int id = 0; id < current_step + 1; id++)
            {
                if (id >= seenStanceTargets[nIndex].agent_state_ids.Count)
                {
                    seenStanceTargets[nIndex].agent_state_ids.Add(-1);
                }
                if (seenStanceTargets[nIndex].agent_state_ids[id] >= 0)
                    CopyState(state_ids[id], seenStanceTargets[nIndex].agent_state_ids[id]);
                else
                    seenStanceTargets[nIndex].agent_state_ids[id] = CopyState(state_ids[id]);
                
                seenStanceTargets[nIndex].valid_state_counter++;
            }
            seenStanceTargets[nIndex].value = val;
            
            for (int b = 0; b < 4; b++)
            {
                seenStanceTargets[nIndex].targetHoldIds[b] = target_hold_ids[b];
            }
            seenStanceTargets[nIndex].stance_id = init_stance_id;
        }
        return;
    }
    
    void AddRootStanceToList()
    {
        if (graphNodes.Count > 0 && graphNodes[0].humanoidStateIds.Count > 0 && graphNodes[0].humanoidStateIds[0] > -1)
        {
            AddSeenStance(graphNodes[0].humanoidStateIds[0], true);
        }
        return;
    }

    public void AddSeenStance(int state_id, bool flag_add_stance)
    {
        if (state_id >= 0 && flag_add_stance)
        {
            int current_stance_id = ReturnNodeId(savedStates[state_id].current_hold_ids);

            if (!(savedStates[state_id].current_hold_ids[2] > -1 || savedStates[state_id].current_hold_ids[3] > -1 || seenStanceCounter == 0))
            {
                return;
            }
            float stance_value = graphNodes[current_stance_id].GetNodeValue();
            bool flag_updated = false;
            for (int t = 0; t < seenStanceCounter && !flag_updated; t++)
            {
                if (seenStances[t].stance_id == current_stance_id)
                {
                    if (seenStances[t].agent_state_id >= 0)
                        CopyState(state_id, seenStances[t].agent_state_id);
                    else
                        seenStances[t].agent_state_id = CopyState(state_id);
                    seenStances[t].stance_value = stance_value;
                    flag_updated = true;
                }
            }

            if (!flag_updated)
            {
                int nIndex = seenStances.Count - 1;
                if (seenStanceCounter < seenStances.Count)
                {
                    nIndex = seenStanceCounter;
                    seenStanceCounter++;
                }
                else
                {
                    seenStances.Add(new ClimberStance());
                    seenStanceCounter = seenStances.Count;
                    nIndex = seenStances.Count - 1;
                }
                
                if (seenStances[nIndex].agent_state_id >= 0)
                    CopyState(state_id, seenStances[nIndex].agent_state_id);
                else
                    seenStances[nIndex].agent_state_id = CopyState(state_id);

                seenStances[nIndex].stance_value = stance_value;
                seenStances[nIndex].stance_id = current_stance_id;
            }
        }
        return;
    }

    int CopyState(int _fromStateId)
    {
        if (_fromStateId < 0)
            return -1;

        int nStateID = GetNextFreeState(savedStates[_fromStateId].pos.Length, savedStates[_fromStateId].spline_init_values.Length);

        CopyState(_fromStateId, nStateID);

        return nStateID;
    }

    int CopyState(int _fromStateId, int _toStateId)
    {
        if (_fromStateId < 0)
            return -1;
        HumanoidBodyState _fState = savedStates[_fromStateId];

        int boneCount = _fState.pos.Length;
        int nStateID = _toStateId;

        HumanoidBodyState _cState = savedStates[nStateID];
        _cState.current_accumulative_reward = _fState.current_accumulative_reward;
        _cState.current_step = _fState.current_step;
        
        for (int i = 0; i < _fState.spline_init_values.Length; i++)
        {
            _cState.spline_init_values[i][0] = _fState.spline_init_values[i][0];
            _cState.spline_init_values[i][1] = _fState.spline_init_values[i][1];
        }
        
        for (int b = 0; b < 4; b++)
        {
            _cState.current_hold_ids[b] = _fState.current_hold_ids[b];

            _cState.current_hold_pos[b] = _fState.current_hold_pos[b];

            _cState.connectorTouchingGround[b] = _fState.connectorTouchingGround[b];
            _cState.connectorTouchingWall[b] = _fState.connectorTouchingWall[b];
            _cState.connectorTouchingHold[b] = _fState.connectorTouchingHold[b];

            _cState.connectorPos[b] = _fState.connectorPos[b];
            _cState.connectorRot[b] = _fState.connectorRot[b];
            _cState.connectorVel[b] = _fState.connectorVel[b];
            _cState.connectorAVel[b] = _fState.connectorAVel[b];
        }

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

    int worker_id = 0;
    void ResetTrajecotriesIfNecessary()
    {
        worker_id = this.base_work_id;
        string update_path = Path.GetFullPath(".") + "/../python/update_frequency" + worker_id.ToString() + ".txt";
        if (File.Exists(update_path))
        {
            StreamReader streamReader = new StreamReader(update_path);
            int stepUpdate = int.Parse(streamReader.ReadLine());
            streamReader.Close();

            File.Delete(update_path);

            int cStep = stepUpdate;
            if (cStep != loggedStep)
            {
                WriteStatistics(cStep);
                
                if (flag_reset_trajectories)
                {
                    seenStanceCounter = 0;
                    seenStanceTargetCounter = 0;
                    AddRootStanceToList();
                    numUpdatedPolicy++;
                }
            }
        }
    }

    private void OpenStatisticsFile()
    {
        if (streamWriter == null && isWorkerIDSet)
        {
            int i = 0;

            string worker_id_base_file_name = base_file_name + worker_id.ToString() + "_";

            while (File.Exists(worker_id_base_file_name + i.ToString() + ".txt"))
            {
                i++;
            }

            string file_name = worker_id_base_file_name + i.ToString() + ".txt";
            if (File.Exists(file_name))
                streamWriter = new StreamWriter(file_name, true);
            else
                streamWriter = new StreamWriter(file_name);
        }
    }

    private void WriteStatistics(int cStep)
    {
        if (streamWriter == null)
        {
            OpenStatisticsFile();
        }

        if (streamWriter == null)
        {
            return;
        }

        if (nonForkedEpisodeCounter > 0 || nonForkedTransitionCounter > 0)
        {
            Debug.Log(cStep.ToString() + ","
                + ((float)nonForked_accumulative_reward / (float)(nonForkedEpisodeCounter + 1e-6)).ToString() + ","
                + ((float)nonForkedTransition_accumulative_reward / (float)(nonForkedTransitionCounter + 1e-6)).ToString() + ","
                + ((float)nonForkedTransition_EpisodeLength / (float)(nonForkedTransitionCounter + 1e-6)).ToString() + ","
                + ((float)nonForkedTransition_NumSpline / (float)(nonForkedTransitionCounter + 1e-6)).ToString() + ","
                + graphNodes.Count.ToString());

            streamWriter.WriteLine(cStep.ToString() + "," 
                + ((float)nonForked_accumulative_reward / (float)(nonForkedEpisodeCounter + 1e-6)).ToString() + ","
                + ((float)nonForkedTransition_accumulative_reward / (float)(nonForkedTransitionCounter + 1e-6)).ToString() + ","
                + ((float)nonForkedTransition_EpisodeLength / (float)(nonForkedTransitionCounter + 1e-6)).ToString() + ","
                + ((float)nonForkedTransition_NumSpline / (float)(nonForkedTransitionCounter + 1e-6)).ToString() + ","
                + graphNodes.Count.ToString());
            streamWriter.Flush();

            nonForked_accumulative_reward = 0;
            nonForkedEpisodeCounter = 0;
            
            nonForkedTransitionCounter = 0;
            nonForkedTransition_accumulative_reward = 0.0f;
            nonForkedTransition_EpisodeLength = 0;
            nonForkedTransition_NumSpline = 0;

            loggedStep = cStep;
        }
        return;
    }

    private void OnApplicationQuit()
    {
        if (streamWriter != null)
            streamWriter.Close();
        
        //int index = 0;

        //while (File.Exists(base_file_name + "pos_reward_" + index.ToString() + ".txt"))
        //{
        //    index++;
        //}

        //StreamWriter statistics_reward_Writer = new StreamWriter(base_file_name + "pos_reward_" + index.ToString() + ".txt");
        //for (int row = 0; row < visited_pos_size_y; row++)
        //{
        //    string line_string = "";
        //    for (int col = 0; col < visited_pos_size_x; col++)
        //    {
        //        if (col == visited_pos_size_x - 1)
        //        {
        //            line_string += (visited_pos_r[col][row]).ToString();
        //        }
        //        else
        //        {
        //            line_string += (visited_pos_r[col][row]).ToString() + ",";
        //        }
        //    }
        //    statistics_reward_Writer.WriteLine(line_string);
        //}
        //statistics_reward_Writer.Flush();
        //statistics_reward_Writer.Close();

        //StreamWriter statistics_count_Writer = new StreamWriter(base_file_name + "pos_count_" + index.ToString() + ".txt");
        //for (int row = 0; row < visited_pos_size_y; row++)
        //{
        //    string line_string = "";
        //    for (int col = 0; col < visited_pos_size_x; col++)
        //    {
        //        if (col == visited_pos_size_x - 1)
        //        {
        //            line_string += (visited_pos_c[col][row]).ToString();
        //        }
        //        else
        //        {
        //            line_string += (visited_pos_c[col][row]).ToString() + ",";
        //        }
        //    }
        //    statistics_count_Writer.WriteLine(line_string);
        //}
        //statistics_count_Writer.Flush();
        //statistics_count_Writer.Close();

        //StreamWriter statistics_success_Writer = new StreamWriter(base_file_name + "pos_success_" + index.ToString() + ".txt");
        //for (int row = 0; row < visited_pos_size_y; row++)
        //{
        //    string line_string = "";
        //    for (int col = 0; col < visited_pos_size_x; col++)
        //    {
        //        if (col == visited_pos_size_x - 1)
        //        {
        //            line_string += (visited_pos_s[col][row]).ToString();
        //        }
        //        else
        //        {
        //            line_string += (visited_pos_s[col][row]).ToString() + ",";
        //        }
        //    }
        //    statistics_success_Writer.WriteLine(line_string);
        //}
        //statistics_success_Writer.Flush();
        //statistics_success_Writer.Close();

        return;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    public void UpdateHumanoidState(int graphId, int _trajectoryIdx)
    {
        if (_trajectoryIdx > -1)
        {
            for (int i = 0; i < 4; i++)
            {
                if (graphNodes[graphId].holdIds[i] != _ControlRigs[_trajectoryIdx].GetCurrentHoldID(i))
                {
                    return;
                }
            }

            int savedNodeIdx = -1;
            while (graphNodes[graphId].humanoidStateIds.Count <= _trajectoryIdx)
            {
                graphNodes[graphId].humanoidStateIds.Add(-1);
            }

            bool flag_update_state = true;
            if (graphNodes[graphId].humanoidStateIds[_trajectoryIdx] < 0)
            {
                graphNodes[graphId].humanoidStateIds[_trajectoryIdx] = GetNextFreeState(_ControlRigs[_trajectoryIdx].jdController.bodyPartsList.Count, _ControlRigs[_trajectoryIdx].spline.Count);
            }
            else if (graphNodes[graphId].nodeIdx == 0)
            {
                flag_update_state = false;
            }
            if (flag_update_state)
            {
                savedNodeIdx = graphNodes[graphId].humanoidStateIds[_trajectoryIdx];
                _ControlRigs[_trajectoryIdx].SaveState(savedNodeIdx);
                // update hold ids around a stance node
                UpdateHoldIdsAroundStanceNodeGivenHipLocation(graphNodes[graphId].nodeIdx, savedStates[savedNodeIdx].pos[1]); // 1 is for chest position

                int c = 0;
                for (int i = 0; i < graphNodes[graphId].humanoidStateIds.Count; i++)
                {
                    if (graphNodes[graphId].humanoidStateIds[i] >= 0)
                    {
                        graphNodes[graphId].hipPos += savedStates[graphNodes[graphId].humanoidStateIds[i]].pos[0];
                        c++;
                    }
                }

                graphNodes[graphId].hipPos /= (c + 1e-6f);
            }
            if (_ControlRigs[_trajectoryIdx].IsTaskOnTheLadderClimbing())
            {
                graphNodes[graphId].planning_index_on_ladder = _ControlRigs[_trajectoryIdx].GetCurrentLimbPlannedIndex();
                if (bestNodeIndex < graphNodes[graphId].nodeIdx)
                {
                    bestNodeIndex = graphNodes[graphId].nodeIdx;
                }
                if (!path_to_target.Contains(graphNodes[graphId].nodeIdx))
                    path_to_target.Add(graphNodes[graphId].nodeIdx);
            }
            //UpdateBestNodeToGoal(nNode);
        }

        return;
    }

    public void CompleteGraphGradually(int count_limb_movement)
    {
        if (!(current_training_type == TrainingType.GraphPSITraining ||
            current_training_type == TrainingType.GraphRandomTraining ||
            current_training_type == TrainingType.GraphTreeTraining))
            return;

        if (to_fix_graph_nodes.Count > 0)
        {
            int graph_node_index = to_fix_graph_nodes[0];
            to_fix_graph_nodes.RemoveAt(0);

            if (!graphNodes[graph_node_index].isFixedNode)
            {
                int na = 0;
                while (na < graphNodes[graph_node_index].possible_actions[count_limb_movement].Count)
                {
                    if (graphNodes[graph_node_index].successRate[count_limb_movement][na] > 0.75f 
                        && graphNodes[graph_node_index].counter_fixed_discovery[count_limb_movement][na] >= 10)
                    {
                        int[] nHoldIDs = graphNodes[graph_node_index].GetTargetHoldIDs(count_limb_movement, na);

                        graphNodes[graph_node_index].possible_actions[count_limb_movement].RemoveAt(na);
                        graphNodes[graph_node_index].successRate[count_limb_movement].RemoveAt(na);
                        graphNodes[graph_node_index].counterSeenAction[count_limb_movement].RemoveAt(na);
                        graphNodes[graph_node_index].preSuccessRate[count_limb_movement].RemoveAt(na);
                        graphNodes[graph_node_index].counter_fixed_discovery[count_limb_movement].RemoveAt(na);

                        AddFixedChild(graph_node_index, nHoldIDs);
                    }
                    else
                    {
                        graphNodes[graph_node_index].counterSeenAction[count_limb_movement][na] = 0;
                        graphNodes[graph_node_index].counter_fixed_discovery[count_limb_movement][na] = 0;
                        na++;
                    }
                }
                
                graphNodes[graph_node_index].isFixedNode = true;
            }
            
            _MaxNodeCount = graphNodes.Count;
        }

        return;
    }

    public class StanceTargetComparer : IComparer<ClimberStanceTarget>
    {
        public int Compare(ClimberStanceTarget x, ClimberStanceTarget y)
        {
            if (x == null)
                return 1;
            if (y == null)
                return -1;
            if (x.value > y.value)
                return -1;
            if (y.value > x.value)
                return 1;
            // TODO: Handle x or y being null, or them not having names
            return 0;
        }
    }

    public class StanceComparer : IComparer<ClimberStance>
    {
        public int Compare(ClimberStance x, ClimberStance y)
        {
            if (x == null)
                return 1;
            if (y == null)
                return -1;
            if (x.stance_value > y.stance_value)
                return -1;
            if (y.stance_value > x.stance_value)
                return 1;
            // TODO: Handle x or y being null, or them not having names
            return 0;
        }
    }

    // for a stationary env, we can create a tree
    public Vector2Int GetRandomState(int _trajectoryIdx, bool flag_test_performance, ref int[] planned_target_ids)
    {
        for (int b = 0; b < 4; b++)
        {
            planned_target_ids[b] = -1;
        }

        float beta = 0.25f;
        int graphNodeID = 0;

        if (current_training_type == TrainingType.UniformTraining)
        {
            int index_seenStances = Random.Range(0, seenStanceCounter);
            if (index_seenStances < seenStances.Count && index_seenStances < seenStanceCounter
                && seenStances[index_seenStances].agent_state_id > -1)
            {
                return new Vector2Int(seenStances[index_seenStances].agent_state_id, 0);
            }

            return new Vector2Int(graphNodes[0].humanoidStateIds[_trajectoryIdx], 0);
        }
        else if (current_training_type == TrainingType.PSITargetTraining)
        {
            if (!flag_test_performance)
            {
                int c_saved_stance_counter = seenStanceTargetCounter;
                if (c_saved_stance_counter > 0 && Random.Range(0, 1f) > beta)
                {
                    // select a trajectory to fork and forking timestep among the best ones
                    int nBest = (int)(beta * c_saved_stance_counter);

                    ClimberStanceTarget[] copyArray = new ClimberStanceTarget[c_saved_stance_counter];
                    seenStanceTargets.CopyTo(0, copyArray, 0, c_saved_stance_counter);
                    System.Array.Sort(copyArray, 0, c_saved_stance_counter, new StanceTargetComparer());

                    int updatingTrajectoryIdx = Random.Range(0, nBest + 1);

                    ClimberStanceTarget forkedStanceTarget = copyArray[updatingTrajectoryIdx];

                    int forkedStep = Random.Range(0, forkedStanceTarget.valid_state_counter);
                    if (forkedStanceTarget.agent_state_ids[forkedStep] > -1)
                    {
                        for (int b = 0; b < 4; b++)
                        {
                            planned_target_ids[b] = forkedStanceTarget.targetHoldIds[b];
                        }
                        return new Vector2Int(forkedStanceTarget.agent_state_ids[forkedStep], 1);
                    }
                }
            }
            // uniform on the seen states
            int index_seenStances = Random.Range(0, seenStanceCounter);
            if (index_seenStances < seenStances.Count && index_seenStances < seenStanceCounter
                && seenStances[index_seenStances].agent_state_id > -1)
            {
                return new Vector2Int(seenStances[index_seenStances].agent_state_id, 0);
            }

            return new Vector2Int(graphNodes[0].humanoidStateIds[_trajectoryIdx], 0);
        }
        else if (current_training_type == TrainingType.GraphPSITraining || current_training_type == TrainingType.PSITraining)
        {
            if (!flag_test_performance)
            {
                int c_saved_stance_counter = seenStanceCounter;
                if (c_saved_stance_counter > 0 && Random.Range(0, 1f) > beta)
                {
                    // select a trajectory to fork and forking timestep among the best ones
                    int nBest = (int)(beta * c_saved_stance_counter);
                    
                    ClimberStance[] copyArray = new ClimberStance[c_saved_stance_counter];
                    seenStances.CopyTo(0, copyArray, 0, c_saved_stance_counter);
                    System.Array.Sort(copyArray, 0, c_saved_stance_counter, new StanceComparer());

                    int updatingTrajectoryIdx = Random.Range(0, nBest + 1);

                    ClimberStance forkedStance = copyArray[updatingTrajectoryIdx];

                    if (forkedStance.agent_state_id > -1)
                    {
                        return new Vector2Int(forkedStance.agent_state_id, 1);
                    }
                }
            }
            // uniform on the seen states
            int index_seenStances = Random.Range(0, seenStanceCounter);
            if (index_seenStances < seenStances.Count && index_seenStances < seenStanceCounter
                && seenStances[index_seenStances].agent_state_id > -1)
            {
                return new Vector2Int(seenStances[index_seenStances].agent_state_id, 0);
            }

            return new Vector2Int(graphNodes[0].humanoidStateIds[_trajectoryIdx], 0);
            
        }
        else if (current_training_type == TrainingType.RandomTraining || current_training_type == TrainingType.GraphRandomTraining)
        {
            // if we are doing random training, we alwayse start from T-Pose
            graphNodeID = 0;
        }
        else if (current_training_type == TrainingType.LadderClimbing)
        {
            int chosen_index = -1;
            for (int i = 0; i < 30 && chosen_index == -1; i++)
            {
                int idx_node = Random.Range((int)0, graphNodes.Count);
                if (graphNodes[idx_node].humanoidStateIds.Count > _trajectoryIdx
                    && graphNodes[idx_node].humanoidStateIds[_trajectoryIdx] > -1)
                {
                    if (chosen_index < 0)
                    {
                        chosen_index = idx_node;
                    }
                }
            }
            if (chosen_index > -1)
            {
                graphNodeID = graphNodes[chosen_index].nodeIdx;
            }
        }
        else if (current_training_type == TrainingType.GraphTreeTraining)
        {
            if (path_to_target.Count > 0)
            {
                int idx_in_path = Random.Range((int)0, path_to_target.Count);
                if (graphNodes[path_to_target[idx_in_path]].humanoidStateIds.Count > _trajectoryIdx
                    && graphNodes[path_to_target[idx_in_path]].humanoidStateIds[_trajectoryIdx] > -1)
                {
                    graphNodeID = path_to_target[idx_in_path];
                }
            }
        }

        // whether if we are training for ladder climbing or branching a tree,
        // we need to beta percent of times start from T-pose and do random transition
        if (Random.Range(0, 1f) > beta)
        {
            if (_trajectoryIdx < graphNodes[graphNodeID].humanoidStateIds.Count
            && graphNodes[graphNodeID].humanoidStateIds[_trajectoryIdx] > -1)
            {
                return new Vector2Int(graphNodes[graphNodeID].humanoidStateIds[_trajectoryIdx], 1);
            }
        }
        
        return new Vector2Int(graphNodes[0].humanoidStateIds[_trajectoryIdx], 0);
    }
    
    public Vector3 GetInitHoldPosition(int hold_id)
    {
        if (mContext == null)
        {
            mContext = new ContextManager(instantObject);
        }
        return mContext.HoldPosition(hold_id);
    }

    public int GetNextFreeState(int boneCount, int splineInitCount)
    {
        savedStates.Add(new HumanoidBodyState(boneCount, splineInitCount));
        return savedStates.Count - 1;
    }

    public void CompleteNode(int r_idx_node, int plan_count_limb_move)
    {
        int counter = 0;
        do
        {
            graphNodes[r_idx_node].CompleteActionList();
            counter++;
        } while (graphNodes[r_idx_node].possible_actions[plan_count_limb_move].Count == 0 && counter < 5);

        return;
    }

    void UpdateHoldIdsAroundStanceNodeGivenHipLocation(int cNodeId, Vector3 hipLocation)
    {
        if (mContext == null)
        {
            mContext = new ContextManager(instantObject);
        }

        float radius = 3.5f;
        
        for (int h = 0; h < mContext.NumHolds(); h++)
        {
            Vector3 dir = (mContext.HoldPosition(h) - hipLocation);
            if (dir.magnitude < radius)
            {
                float theta = Mathf.Atan2(dir.normalized[1], dir.normalized[0]);
                if (theta >= 0 && theta <= Mathf.PI / 2f)
                {
                    graphNodes[cNodeId].UpdateHoldIdsForLimb(3, h);
                }
                if (theta >= Mathf.PI / 2f && theta <= Mathf.PI)
                {
                    graphNodes[cNodeId].UpdateHoldIdsForLimb(2, h);
                }
                if (theta >= -Mathf.PI / 2f && theta <= 0)
                {
                    graphNodes[cNodeId].UpdateHoldIdsForLimb(1, h);
                }
                if (theta >= -Mathf.PI && theta <= -Mathf.PI / 2f)
                {
                    graphNodes[cNodeId].UpdateHoldIdsForLimb(0, h);
                }
            }
        }

        return;
    }

    void UpdateHoldIdsAroundStanceNodeGivenRefPoint(Vector2Int ref_point, int cNodeId, int limb_id)
    {
        if (ref_point[0] == -1)
        {
            int bias = limb_id <= 1 ? 0 : 1;
            int limb_to_hold = limb_id <= 1 ? limb_id : (int)((limb_id + 1) / 2) - 1;
            graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, limb_to_hold + 1 + 4 * bias);
        }
        else
        {
            int col = -1;
            int row = -1;
            if (graphNodes[cNodeId].holdIds[limb_id] == -1)
            {
                int ref_col = ref_point[1] % 4;
                int ref_row = ref_point[1] / 4;
                if (ref_point[0] == 2 && ref_col < 3)
                    ref_col++;
                
                switch (limb_id)
                {
                    case 0:
                        col = ref_col > 0 ? ref_col - 1 : ref_col;
                        row = ref_row > 0 ? ref_row - 1 : ref_row;
                        break;
                    case 1:
                        col = ref_col;
                        row = ref_row > 0 ? ref_row - 1 : ref_row;
                        break;
                    case 2:
                        col = ref_col > 0 ? ref_col - 1 : ref_col;
                        row = ref_row;
                        break;
                    case 3:
                        col = ref_col;
                        row = ref_row;
                        break;
                }

            }
            else
            {
                col = graphNodes[cNodeId].holdIds[limb_id] % 4;
                row = graphNodes[cNodeId].holdIds[limb_id] / 4;
            }

            // free limb
            graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, -1);
            // add neighbour hold
            int hold_id = row * 4 + col;
            
            graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, hold_id);
            if (col > 0)
                graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, hold_id - 1);
            if (col < 3)
                graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, hold_id + 1);
            if (row > 0)
                graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, hold_id - 4);
            if (row < 3)
                graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, hold_id + 4);

            if (col > 0 && row > 0)
                graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, hold_id - 1 - 4);
            if (col > 0 && row < 3)
                graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, hold_id - 1 + 4);
            if (col < 3 && row > 0)
                graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, hold_id + 1 - 4);
            if (col < 3 && row < 3)
                graphNodes[cNodeId].UpdateHoldIdsForLimb(limb_id, hold_id + 1 + 4);
        }

        return;
    }
    
    void UpdateTransitionStatistics(int _fromNode, int _actionIndex, int plan_count_limb_movement)
    {
        if (!(current_training_type == TrainingType.GraphPSITraining ||
            current_training_type == TrainingType.GraphRandomTraining ||
            current_training_type == TrainingType.GraphTreeTraining))
            return;
        //some statistics on success and failure
        if (graphNodes[_fromNode].counterSeenAction[plan_count_limb_movement][_actionIndex] > 20)
        {
            if (Mathf.Abs(graphNodes[_fromNode].preSuccessRate[plan_count_limb_movement][_actionIndex] 
                - graphNodes[_fromNode].successRate[plan_count_limb_movement][_actionIndex]) < 0.1f)
            {
                graphNodes[_fromNode].counter_fixed_discovery[plan_count_limb_movement][_actionIndex]++;
                if (graphNodes[_fromNode].counter_fixed_discovery[plan_count_limb_movement][_actionIndex] >= 10)
                {
                    graphNodes[_fromNode].counter_fixed_discovery[plan_count_limb_movement][_actionIndex] = 10;

                    if (graphNodes[_fromNode].successRate[plan_count_limb_movement][_actionIndex] > 0.75f)
                    {
                        graphNodes[_fromNode].isFixedNode = false;
                        to_fix_graph_nodes.Add(_fromNode);
                    }
                }
            }
            else
            {
                graphNodes[_fromNode].counter_fixed_discovery[plan_count_limb_movement][_actionIndex] = 0;
            }
            graphNodes[_fromNode].counterSeenAction[plan_count_limb_movement][_actionIndex] = 0;
        }

        return;
    }

    public void AddFailure(int _fromNode, int _actionIndex, int plan_count_limb_movement)
    {
        if (!(current_training_type == TrainingType.GraphPSITraining ||
            current_training_type == TrainingType.GraphRandomTraining ||
            current_training_type == TrainingType.GraphTreeTraining))
            return;

        if (_fromNode != -1)
        {
            if (_actionIndex > -1 && _actionIndex < graphNodes[_fromNode].counterSeenAction[plan_count_limb_movement].Count)
            {
                if (graphNodes[_fromNode].counterSeenAction[plan_count_limb_movement][_actionIndex] == 0)
                {
                    graphNodes[_fromNode].preSuccessRate[plan_count_limb_movement][_actionIndex] = graphNodes[_fromNode].successRate[plan_count_limb_movement][_actionIndex];
                }

                graphNodes[_fromNode].successRate[plan_count_limb_movement][_actionIndex] -= minSuccessRateChange[plan_count_limb_movement];
                graphNodes[_fromNode].successRate[plan_count_limb_movement][_actionIndex] 
                    = Mathf.Max(minSuccessRate, graphNodes[_fromNode].successRate[plan_count_limb_movement][_actionIndex]);
                
                graphNodes[_fromNode].counterSeenAction[plan_count_limb_movement][_actionIndex]++;

                UpdateTransitionStatistics(_fromNode, _actionIndex, plan_count_limb_movement);
            }
        }

        return;
    }
    
    public void AddSuccess(int _fromNode, int _actionIndex, int plan_count_limb_movement)
    {
        if (!(current_training_type == TrainingType.GraphPSITraining ||
            current_training_type == TrainingType.GraphRandomTraining ||
            current_training_type == TrainingType.GraphTreeTraining))
            return;

        if (_fromNode != -1)
        {
            if (_actionIndex > -1 && _actionIndex < graphNodes[_fromNode].counterSeenAction[plan_count_limb_movement].Count)
            {
                if (graphNodes[_fromNode].counterSeenAction[plan_count_limb_movement][_actionIndex] == 0)
                {
                    graphNodes[_fromNode].preSuccessRate[plan_count_limb_movement][_actionIndex] = graphNodes[_fromNode].successRate[plan_count_limb_movement][_actionIndex];
                }

                graphNodes[_fromNode].successRate[plan_count_limb_movement][_actionIndex] += minSuccessRateChange[plan_count_limb_movement];
                graphNodes[_fromNode].successRate[plan_count_limb_movement][_actionIndex] 
                    = Mathf.Min(1.0f, graphNodes[_fromNode].successRate[plan_count_limb_movement][_actionIndex]);

                graphNodes[_fromNode].counterSeenAction[plan_count_limb_movement][_actionIndex]++;

                UpdateTransitionStatistics(_fromNode, _actionIndex, plan_count_limb_movement);
            }
        }

        return;
    }

    public static ulong StanceToKey(int[] holdIds)
    {
        ulong result = 0;
	    for (int i = 0; i < 4; i++)
	    {
		    uint uNode = (uint)(holdIds[i] + 1);
            result = (result+uNode);
		    if (i < 3)
                result = result << 8; //shift, assuming max 256 holds
	    }
	    return result;
    }

    public int ReturnNodeId(int[] holdIds)
    {
        ulong key = StanceToKey(holdIds);
        if (graphNodeDict.ContainsKey(key))
        {
            int search_node_index = graphNodeDict[key];

            for (int h = 0; h < 4; h++)
            {
                if (graphNodes[search_node_index].holdIds[h] != holdIds[h])
                {
                    return -1;
                }
            }

            return search_node_index;            
        }
        return -1;
    }

    public float GetCostToNode(int starting_index)
    {
        int c_node_index = starting_index;
        int counter = 0;
        float v = 0;
        while (c_node_index != -1 && counter < graphNodes.Count + 1)
        {
            v += graphNodes[c_node_index].costToNode;
            c_node_index = graphNodes[c_node_index].bestParentNodeIdx;
            counter++;
        }
        if (counter >= graphNodes.Count && c_node_index != -1)
        {
            return float.MaxValue;
        }

        return v;
    }

    public Vector2Int GetNextAction(int[] curHoldIds)
    {
        Vector2Int ret_val = new Vector2Int(-1, -1);
        int c_node_index = -1;
        int t_node_index = -1;
        if (path_to_target.Count > 0)
        {
            for (int i = path_to_target.Count - 1; i >= 0; i--)
            {
                if (Tools.IsSetAEqualSetB(curHoldIds, graphNodes[path_to_target[i]].holdIds))
                {
                    c_node_index = path_to_target[i];
                    if (i - 1 >= 0)
                    {
                        t_node_index = path_to_target[i - 1];
                        break;
                    }
                }
            }
        }

        if (c_node_index >= 0 && t_node_index >= 0)
        {
            for (int h = 0; h < 4; h++)
            {
                if (graphNodes[c_node_index].holdIds[h] != graphNodes[t_node_index].holdIds[h])
                {
                    ret_val[0] = h;
                    ret_val[1] = graphNodes[t_node_index].holdIds[h];

                    return ret_val;
                }
            }
        }
        return ret_val;
    }

    public static bool IsValidTransition(int[] _fromStance, int[] _toStance)
    {
        if (!IsValidStance(_fromStance))
            return false;
        if (!IsValidStance(_toStance))
            return false;

        // action cannot put the agent in the same stance
        bool isStanceTheSame = true;
        for (int i = 0; i < 4; i++)
        {
            if (_fromStance[i] != _toStance[i])
            {
                isStanceTheSame = false;
            }
        }
        if (isStanceTheSame)
            return false;

        int current_count_limb_movment = Tools.GetDiffBtwSetASetB(_fromStance, _toStance) - 1;
        if (current_count_limb_movment < min_count_limb_movement - 1 || current_count_limb_movment > max_count_limb_movement - 1)
        {
            return false;
        }

        if (_fromStance[2] < 0 && _fromStance[3] < 0)
        {
            if (_toStance[2] >= 0 || _toStance[3] >= 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            if (_fromStance[2] >= 0 && _fromStance[3] >= 0)
            {
                if (_toStance[2] == _fromStance[2] || _toStance[3] == _fromStance[3])
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                int connected_hand = -1;
                if (_fromStance[2] >= 0)
                    connected_hand = 2;
                else if (_fromStance[3] >= 0)
                    connected_hand = 3;
                if (_toStance[connected_hand] == _fromStance[connected_hand])
                    return true;
                else
                    return false;
            }
        }
        
        return true;
    }

    public static bool IsValidStance(int[] curHoldIds)
    {
        if (curHoldIds[0] < 0 && curHoldIds[1] < 0 && curHoldIds[2] < 0 && curHoldIds[3] < 0)
        {
            return true;
        }

        if (curHoldIds[2] < 0 && curHoldIds[3] < 0)
        {
            return false;
        }

        int[] row_from_ids = { -1, -1, -1, -1 };
        int[] col_from_ids = { -1, -1, -1, -1 };
        for (int i = 0; i < 4; i++)
        {
            if (curHoldIds[i] >= 0)
            {
                row_from_ids[i] = (int)(curHoldIds[i] / 4);
                col_from_ids[i] = (int)(curHoldIds[i] % 4);
            }

            if (row_from_ids[i] >= 4 || col_from_ids[i] >= 4)
            {
                return false;
            }
        }

        float hand_hand_dis = 0.0f;
        if (row_from_ids[2] >= 0 && row_from_ids[3] >= 0)
        {
            hand_hand_dis = (new Vector2(row_from_ids[2], col_from_ids[2]) - new Vector2(row_from_ids[3], col_from_ids[3])).magnitude;
        }

        float leg_leg_dis = 0.0f;
        if (row_from_ids[0] >= 0 && row_from_ids[1] >= 0)
        {
            leg_leg_dis = (new Vector2(row_from_ids[0], col_from_ids[0]) - new Vector2(row_from_ids[1], col_from_ids[1])).magnitude;
        }

        if (hand_hand_dis >= 2.95f || leg_leg_dis >= 2.95f)
        {
            return false;
        }

        float hand_leg_dis = 0.0f;
        for (int h = 2; h < 4; h++)
        {
            if (row_from_ids[h] >= 0)
            {
                Vector2 hand_p = new Vector2(row_from_ids[h], col_from_ids[h]);
                for (int l = 0; l < 2; l++)
                {
                    if (row_from_ids[l] >= 0)
                    {
                        Vector2 leg_p = new Vector2(row_from_ids[l], col_from_ids[l]);
                        float cDis = (hand_p - leg_p).magnitude;
                        if (hand_leg_dis < cDis)
                        {
                            hand_leg_dis = cDis;
                        }
                    }
                }
            }
        }

        if (hand_leg_dis >= 2.95f)
        {
            return false;
        }
        return true;
    }

    void AddFixedChild(int _fatherID, int[] holdIds)
    {
        if (_fatherID < 0)
            return;

        int found_node_idx = ReturnNodeId(holdIds);
        if (found_node_idx == -1)
        {
            found_node_idx = AddNode(_fatherID, holdIds, -1, true);
        }

        if (found_node_idx < 0)
            return;

        GraphNode nNode = graphNodes[found_node_idx];

        if (_fatherID > -1)
        {
            int _diff_hold_ids = Tools.GetDiffBtwSetASetB(graphNodes[_fatherID].holdIds, nNode.holdIds) - 1;
            // creating directed graph
            if (_diff_hold_ids >= 0 && _diff_hold_ids <= 1)
            {
                if (!graphNodes[_fatherID].childNodeIds[_diff_hold_ids].Contains(nNode.nodeIdx))
                {
                    graphNodes[_fatherID].childNodeIds[_diff_hold_ids].Add(nNode.nodeIdx);
                }
            }
        }
    }

    public int AddNode(int _fatherID, int[] holdIds, int _trajectoryIdx, bool flag_add_anyway = false)
    {
        int found_node_idx = ReturnNodeId(holdIds);
        if (!flag_add_anyway && !(found_node_idx > -1 && _trajectoryIdx > -1))
        {
            if (!IsValidStance(holdIds) || graphNodes.Count >= _MaxNodeCount )
            {
                return -1;
            }
        }

        if (_fatherID == found_node_idx && found_node_idx >= 0)
        {
            return _fatherID;
        }

        float costToNode = getCostAtNode(holdIds);
        if (_fatherID > -1)
        {
            costToNode += getCostMovementLimbs(graphNodes[_fatherID].holdIds, holdIds);
        }

        GraphNode nNode = null;
        if (found_node_idx == -1)
        {
            graphNodes.Add(new GraphNode());
            nNode = graphNodes[graphNodes.Count - 1];
            nNode.nodeIdx = graphNodes.Count - 1;

            for (int h = 0; h < 4; h++)
            {
                nNode.holdIds[h] = holdIds[h];
            }

            graphNodeDict.Add(StanceToKey(holdIds), nNode.nodeIdx);

            // ref_point is one of the hands
            Vector2Int ref_point = new Vector2Int(-1, -1);
            for (int h = 2; h < 4 && ref_point[0] == -1; h++)
            {
                if (nNode.holdIds[h] > -1)
                {
                    ref_point[0] = h;
                    ref_point[1] = nNode.holdIds[h];
                }
            }

            for (int b = 0; b < 4; b++)
            {
                UpdateHoldIdsAroundStanceNodeGivenRefPoint(ref_point, nNode.nodeIdx, b);
            }
        }
        else
        {
            nNode = graphNodes[found_node_idx];
            nNode.nodeIdx = found_node_idx;
        }
        
        // this part is for updating bestParentIdx
        bool change_parent = false;
        if (_fatherID > -1)
        {
            // node already exists
            if (found_node_idx != -1 && nNode.bestParentNodeIdx > -1)
            {
                int oldFatherIdx = nNode.bestParentNodeIdx;
                float oldCost = GetCostToNode(nNode.nodeIdx);
                float newCost = GetCostToNode(_fatherID) + costToNode;
                //change father case
                if (newCost < oldCost && _fatherID != nNode.bestParentNodeIdx)
                {
                    change_parent = true;
                    // check loop
                    nNode.bestParentNodeIdx = _fatherID;
                    int c_node_index = nNode.nodeIdx;
                    int counter = 0;
                    while (c_node_index != -1 && counter < graphNodes.Count + 1)
                    {
                        c_node_index = graphNodes[c_node_index].bestParentNodeIdx;
                        counter++;
                    }
                    if (counter >= graphNodes.Count && c_node_index != -1)
                    {
                        nNode.bestParentNodeIdx = oldFatherIdx;
                        change_parent = false;
                    }
                }

                if (graphNodes[_fatherID].humanoidStateIds.Count <= 0)
                {
                    change_parent = false;
                }
            }
            else
            {
                change_parent = true;
            }
        }
        else if (nNode.bestParentNodeIdx == -1)
        {
            change_parent = true;
        }

        if (change_parent)
        {
            nNode.bestParentNodeIdx = _fatherID;
            nNode.costToNode = costToNode;
        }
        
        UpdateHumanoidState(nNode.nodeIdx, _trajectoryIdx);

        return nNode.nodeIdx;
    }

    void UpdateBestNodeToGoal(GraphNode nNode)
    {
        return;
        Vector3 midPos = mContext.GetExpectedMidHoldStancePos(nNode.holdIds);
        float cDis = (mContext.GetGoalHold() - midPos).magnitude;
        float cost_to_node = GetCostToNode(nNode.nodeIdx);
        bool change_best_node = false;
        if (disToTarget > cDis)
        {
            disToTarget = cDis;

            change_best_node = true;
        }
        else if (disToTarget - cDis == 0f)
        {
            if (least_cost_to_goal > cost_to_node)
            {
                change_best_node = true;
            }
        }

        if (change_best_node)
        {
            bestNodeIndex = nNode.nodeIdx;

            least_cost_to_goal = cost_to_node;
            disToTarget = cDis;

            path_to_target.Clear();
            int cNodeIdx = bestNodeIndex;
            int counter = 0;
            while (cNodeIdx != -1 && counter < 100)
            {
                path_to_target.Add(cNodeIdx);
                cNodeIdx = graphNodes[cNodeIdx].bestParentNodeIdx;
                counter++;
            }
        }
        return;
    }

    ////////////////////////////////// Academy function ///////////////////////////////////////////
    public override void InitializeAcademy()
    {
        for (int i = 0; i < visited_pos_size_x; i++)
        {
            visited_pos_r[i] = new float[visited_pos_size_y];
            visited_pos_c[i] = new int[visited_pos_size_y];
            visited_pos_s[i] = new int[visited_pos_size_y];
        }

        if (mContext == null)
            mContext = new ContextManager(instantObject);
        
        Monitor.verticalOffset = 1f;

        // We increase the Physics solver iterations in order to
        // make walker joint calculations more accurate.
        Physics.defaultSolverIterations = 6;
        Physics.defaultSolverVelocityIterations = 1;
        Time.fixedDeltaTime = 0.01f; //(75fps). default is .2 (60fps)
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

    //////////////////////////////// Siggraph 2017 Heuristics //////////////////////////////////////
    float getCostAtNode(int[] iCStance)
    {
        if (mContext == null)
            mContext = new ContextManager(instantObject);

        float k_crossing = 100;
        float k_hanging_hand = 200 + 50; // 200
        float k_hanging_leg = 10 + 10; // 10
                                       //		float k_hanging_more_than2 = 0;//100;
        float k_matching = 100;
        float k_dis = 1000;

        float _cost = 0.0f;

        // punish for hanging more than one limb
        int counter_hanging = 0;
        for (int i = 0; i < iCStance.Length; i++)
        {
            if (iCStance[i] == -1)
            {
                counter_hanging++;

                if (i >= 2)
                {
                    // punish for having hanging hand
                    _cost += k_hanging_hand;

                }
                else
                {
                    // punish for having hanging hand
                    _cost += k_hanging_leg;
                }
            }
        }

        _cost += k_hanging_leg * counter_hanging;

        // crossing hands
        if (iCStance[2] != -1 && iCStance[3] != -1)
        {
            Vector3 rHand = mContext.HoldPosition(iCStance[3]);
            Vector3 lHand = mContext.HoldPosition(iCStance[2]);

            if (rHand.x < lHand.x)
            {
                _cost += k_crossing;
            }
        }

        // crossing feet
        if (iCStance[0] != -1 && iCStance[1] != -1)
        {
            Vector3 lLeg = mContext.HoldPosition(iCStance[0]);
            Vector3 rLeg = mContext.HoldPosition(iCStance[1]);

            if (rLeg.x < lLeg.x)
            {
                _cost += k_crossing;
            }
        }

        // crossing hand and foot
        for (int i = 0; i <= 1; i++)
        {
            if (iCStance[i] != -1)
            {
                Vector3 leg = mContext.HoldPosition(iCStance[i]);
                for (int j = 2; j <= 3; j++)
                {
                    if (iCStance[j] != -1)
                    {
                        Vector3 hand = mContext.HoldPosition(iCStance[j]);

                        if (hand.y <= leg.y)
                        {
                            _cost += k_crossing;
                        }
                    }
                }
            }
        }

        //feet matching
        if (iCStance[0] == iCStance[1])
        {
            _cost += k_matching;
        }

        //punishment for hand and leg being close
        for (int i = 0; i <= 1; i++)
        {
            if (iCStance[i] != -1)
            {
                Vector3 leg = mContext.HoldPosition(iCStance[i]);
                for (int j = 2; j <= 3; j++)
                {
                    if (iCStance[j] != -1)
                    {
                        Vector3 hand = mContext.HoldPosition(iCStance[j]);

                        float cDis = (hand - leg).magnitude;

                        const float handAndLegDistanceThreshold = 0.5f;//mClimberSampler->climberRadius / 2.0f;
                        if (cDis < handAndLegDistanceThreshold)
                        {
                            cDis /= handAndLegDistanceThreshold;
                            _cost += k_dis * Mathf.Max(0.0f, 1.0f - cDis);
                        }
                    }
                }
            }
        }

        Vector3 midPoint1 = mContext.GetMidHoldStancePos(iCStance);
        float _dis = (midPoint1 - mContext.GetGoalHold()).magnitude;
        return _cost + 10 * _dis; // _cost
    }
        
    float getCostMovementLimbs(int[] si, int[] sj)
    {
        if (mContext == null)
            mContext = new ContextManager(instantObject);
        float k_dis = 1.0f;
        float k_2limbs = 120.0f;//20.0f;
        float k_pivoting_close_dis = 500.0f;

        //First get the actual distance between holds. We scale it up 
        //as other penalties are not expressed in meters
        float _cost = k_dis * getDisFromStanceToStance(si, sj);

        //penalize moving 2 limbs, except in "ladder climbing", i.e., moving opposite hand and leg
        bool flag_punish_2Limbs = true;
        //bool is2LimbsPunished = false;

        if (Tools.GetDiffBtwSetASetB(si, sj) > 1.0f)
        {

            if (si[0] != sj[0] && si[3] != sj[3] && firstHoldIsLower(si[0], sj[0]))
            {
                flag_punish_2Limbs = false;
                if (sj[0] != -1 && sj[3] != -1 && mContext.HoldPosition(sj[3]).x - mContext.HoldPosition(sj[0]).x < 0.5f)
                    flag_punish_2Limbs = true;
                if (sj[0] != -1 && sj[3] != -1 && mContext.HoldPosition(sj[3]).y - mContext.HoldPosition(sj[0]).y < 0.5f)
                    flag_punish_2Limbs = true;
            }

            if (si[1] != sj[1] && si[2] != sj[2] && firstHoldIsLower(si[1], sj[1]))
            {
                flag_punish_2Limbs = false;
                if (sj[1] != -1 && sj[2] != -1 && mContext.HoldPosition(sj[1]).x - mContext.HoldPosition(sj[2]).x < 0.5f)
                    flag_punish_2Limbs = true;
                if (sj[1] != -1 && sj[2] != -1 && mContext.HoldPosition(sj[2]).y - mContext.HoldPosition(sj[1]).y < 0.5f)
                    flag_punish_2Limbs = true;
            }

            if (flag_punish_2Limbs)
                _cost += k_2limbs;
        }

        // calculating the stance during the transition
        int[] sn = { -1, -1, -1, -1 };
        int count_free_limbs = 0;
        for (int i = 0; i < si.Length; i++)
        {
            if (si[i] != sj[i])
            {
                sn[i] = -1;
                count_free_limbs++;
            }
            else
            {
                sn[i] = si[i];
            }
        }
        // free another
        if (count_free_limbs >= 2 && Tools.GetDiffBtwSetASetB(si, sj) == 1.0f)
            _cost += k_2limbs;

        // punish for pivoting!!!
        float v = 0.0f;
        float max_dis = float.MinValue;
        for (int i = 0; i <= 1; i++)
        {
            if (sn[i] != -1)
            {
                Vector3 leg = mContext.HoldPosition(sn[i]);
                for (int j = 2; j <= 3; j++)
                {
                    if (sn[j] != -1)
                    {
                        Vector3 hand = mContext.HoldPosition(sn[j]);

                        float cDis = (hand - leg).magnitude;

                        if (max_dis < cDis)
                            max_dis = cDis;
                    }
                }
            }
        }
        if (max_dis >= 0 && max_dis < mContext.GetClimberRadius() / 2.0f && count_free_limbs > 1.0f)
        {
            v += k_pivoting_close_dis;
        }
        _cost += v;

        return _cost;
    }

    float getDisFromStanceToStance(int[] si, int[] sj)
    {
        float cCount = 0.0f;
        List<Vector3> hold_points_i = new List<Vector3>();
        mContext.GetHoldStancePosFrom(si, ref hold_points_i, ref cCount);//Vector3 midPoint1 = 

        List<Vector3> hold_points_j = new List<Vector3>();
        mContext.GetHoldStancePosFrom(sj, ref hold_points_j, ref cCount);//Vector3 midPoint2 = 

        float cCost = 0.0f;
        float hangingLimbExpectedMovement = 2.0f;
        for (int i = 0; i < si.Length; i++)
        {
            float coeff_cost = 1.0f;
            if (si[i] != sj[i])
            {
                Vector3 pos_i;
                if (si[i] != -1)
                {
                    pos_i = hold_points_i[i];
                }
                else
                {
                    //pos_i = e_hold_points_i[i];
                    cCost += 0.5f;
                    continue;
                }
                Vector3 pos_j;
                if (sj[i] != -1)
                {
                    pos_j = hold_points_j[i];
                }
                else
                {
                    //pos_j = e_hold_points_j[i];
                    cCost += hangingLimbExpectedMovement;
                    continue;
                }

                //favor moving hands
                if (i >= 2)
                    coeff_cost = 0.9f;

                cCost += coeff_cost * (pos_i - hold_points_j[i]).sqrMagnitude;
            }
            else
            {
                if (sj[i] == -1)
                {
                    cCost += hangingLimbExpectedMovement;
                }
            }
        }

        return Mathf.Sqrt(cCost);
    }

    bool firstHoldIsLower(int hold1, int hold2)
    {
        if (hold1 == -1 && hold2 == -1)
            return false;
        if (hold1 != -1 && mContext.HoldPosition(hold1).y < mContext.HoldPosition(hold2).y)
        {
            return true;
        }
        //first hold is "free" => we can't really know 
        return false;
    }
}
