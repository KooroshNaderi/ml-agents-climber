using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class WalkerAgent : Agent
{
    public WalkerAcademy HighLevelGuide;

    [Header("Specific to Walker")] [Header("Target To Walk Towards")] [Space(10)]
    public Transform target;

    Vector3 dirToTarget;
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
    JointDriveController jdController;
    bool isNewDecisionStep;
    int currentDecisionStep;

    const bool flag_use_early_stopping = false;
    const bool flag_use_psi = true;

    List<int> agent_state_ids = new List<int>();
    bool isTrajectoryForked = false;
    float current_accumulative_reward = 0f;
    int current_step = 0; // index on the current valid saved state
    int starting_step = 0;
    int updatingTrajectoryIdx = -1;

    Vector3 parentPos = new Vector3();
    Quaternion parentRotation = new Quaternion();

    public override void InitializeAgent()
    {
        jdController = GetComponent<JointDriveController>();
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

        SaveState();

        parentPos = this.transform.position;
        parentRotation = this.transform.rotation;
    }

    public override void MyDoneFunc(bool is_failure, bool reset_needed = false)
    {
        // do some stuff before finishing the episode
        if (!is_failure)
        {
            SaveState();
        }
        base.MyDoneFunc(is_failure, reset_needed);
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp)
    {
        var rb = bp.rb;
        AddVectorObs(bp.groundContact.touchingGround ? 1 : 0); // Is this bp touching the ground
        AddVectorObs(rb.velocity);
        AddVectorObs(rb.angularVelocity);
        Vector3 localPosRelToHips = hips.InverseTransformPoint(rb.position);
        AddVectorObs(localPosRelToHips);

        if (bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR &&
            bp.rb.transform != footL && bp.rb.transform != footR && bp.rb.transform != head)
        {
            AddVectorObs(bp.currentXNormalizedRot);
            AddVectorObs(bp.currentYNormalizedRot);
            AddVectorObs(bp.currentZNormalizedRot);
            AddVectorObs(bp.currentStrength / jdController.maxJointForceLimit);
        }
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations()
    {
        jdController.GetCurrentJointForces();

        AddVectorObs(dirToTarget.normalized);
        AddVectorObs(jdController.bodyPartsDict[hips].rb.position);
        AddVectorObs(hips.forward);
        AddVectorObs(hips.up);

        foreach (var bodyPart in jdController.bodyPartsDict.Values)
        {
            CollectObservationBodyPart(bodyPart);
        }
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        dirToTarget = target.position - jdController.bodyPartsDict[hips].rb.position;

        // Apply action to all relevant body parts. 
        if (isNewDecisionStep)
        {
            var bpDict = jdController.bodyPartsDict;
            int i = -1;

            bpDict[chest].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[spine].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);

            bpDict[thighL].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[thighR].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[shinL].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[shinR].SetJointTargetRotation(vectorAction[++i], 0, 0);
            bpDict[footR].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);
            bpDict[footL].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], vectorAction[++i]);


            bpDict[armL].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
            bpDict[armR].SetJointTargetRotation(vectorAction[++i], vectorAction[++i], 0);
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

        IncrementDecisionTimer();

        // Set reward for this step according to mixture of the following elements.
        // a. Velocity alignment with goal direction.
        // b. Rotation alignment with goal direction.
        // c. Encourage head height.
        // d. Discourage head movement.
        AddReward(
            +0.03f * Vector3.Dot(dirToTarget.normalized, jdController.bodyPartsDict[hips].rb.velocity)
            + 0.01f * Vector3.Dot(dirToTarget.normalized, hips.forward)
            + 0.02f * (head.position.y - hips.position.y)
            - 0.01f * Vector3.Distance(jdController.bodyPartsDict[head].rb.velocity,
                jdController.bodyPartsDict[hips].rb.velocity)
        );
    }

    /// <summary>
    /// Only change the joint settings based on decision frequency.
    /// </summary>
    public void IncrementDecisionTimer()
    {
        if (currentDecisionStep == agentParameters.numberOfActionsBetweenDecisions ||
            agentParameters.numberOfActionsBetweenDecisions == 1)
        {
            current_step++;
            if (!SaveState())
                current_step--;
            else
                HighLevelGuide.UpdateTrajectory(updatingTrajectoryIdx, agent_state_ids[current_step], current_accumulative_reward);

            HighLevelGuide.IncreaseSampleCount();

            currentDecisionStep = 1;
            isNewDecisionStep = true;
        }
        else
        {
            currentDecisionStep++;
            isNewDecisionStep = false;
        }

        if (flag_use_early_stopping)
        {
            if (GetStepCount() % 200 == 0 && GetStepCount() > 0)
            {
                float xV = Vector3.Dot(jdController.bodyPartsDict[hips].rb.velocity, new Vector3(1, 0, 0));
                float zV = Vector3.Dot(jdController.bodyPartsDict[hips].rb.velocity, new Vector3(0, 0, 1));
                if (Mathf.Abs(xV) < HighLevelGuide.Threshold * Mathf.Abs(zV))
                {
                    MyDoneFunc(true);
                }
            }
        }
    }

    public bool SaveState()
    {
        if (hips.localPosition.y < -5.0f)
        {
            MyDoneFunc(true);
            return false;
        }

        while (current_step >= agent_state_ids.Count)
        {
            agent_state_ids.Add(-1);
        }

        if (agent_state_ids[current_step] == -1)
        {
            agent_state_ids[current_step] = HighLevelGuide.GetNextFreeState(jdController.bodyPartsList.Count);
        }

        int _stateID = agent_state_ids[current_step];

        WalkerAcademy.HumanoidState _cState = HighLevelGuide.savedStates[_stateID];

        current_accumulative_reward += GetReward();
        _cState.current_accumulative_reward = current_accumulative_reward;
        _cState.current_step = starting_step + current_step;

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

        return true;
    }

    public bool LoadState(float beta)
    {
        if (HighLevelGuide.sortedTrajectoriesIndices.Count <= 0)
        {
            return false;
        }

        this.transform.position = parentPos;
        this.transform.rotation = parentRotation;

        // select a trajectory to fork and forking timestep among the best ones
        int nBest = (int)(beta * HighLevelGuide.sortedTrajectoriesIndices.Count);
        updatingTrajectoryIdx = HighLevelGuide.sortedTrajectoriesIndices[Random.Range(0, nBest + 1)];
        WalkerAcademy.WalkerTrajectory forkedTrajectory = HighLevelGuide.savedTrajectories[updatingTrajectoryIdx];
        int forkTimeStep = Random.Range(0, forkedTrajectory.count_valid_states);
        int forkParentStateIdx = forkedTrajectory.agent_state_ids[forkTimeStep];

        WalkerAcademy.HumanoidState _cState = HighLevelGuide.savedStates[forkParentStateIdx];

        for (int i = 0; i < jdController.bodyPartsList.Count; i++)
        {
            jdController.bodyPartsList[i].rb.transform.localPosition = _cState.pos[i];
            jdController.bodyPartsList[i].rb.transform.localRotation = _cState.rot[i];
            jdController.bodyPartsList[i].rb.velocity = _cState.vel[i];
            jdController.bodyPartsList[i].rb.angularVelocity = _cState.aVel[i];

            jdController.bodyPartsList[i].groundContact.touchingGround = _cState.touchingGround[i];
            jdController.bodyPartsList[i].wallContact.touchingWall = _cState.touchingWall[i];

            jdController.bodyPartsList[i].holdContact.touchingHold = _cState.touchingHold[i];
            jdController.bodyPartsList[i].holdContact.hold_id = _cState.touchingHoldId[i];

            jdController.bodyPartsList[i].targetContact.touchingTarget = _cState.touchingTarget[i];
        }

        starting_step = _cState.current_step;
        return true;
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void AgentReset()
    {
        bool forkToTrajectory = false;
        // update the high-level with current agent's trajectory
        HighLevelGuide.AddTrajectory(agent_state_ids, current_step, starting_step, isTrajectoryForked);

        starting_step = 0;
        current_step = 0;
        current_accumulative_reward = 0f;

        if (flag_use_psi)
        {
            float beta = 0.25f;
            if (HighLevelGuide.savedTrajectories.Count > 0 && Random.Range(0, 1f) > beta)
            {
                forkToTrajectory = LoadState(beta);
            }
        }

        if (!forkToTrajectory)
        {
            if (dirToTarget != Vector3.zero)
            {
                transform.rotation = Quaternion.LookRotation(dirToTarget);
            }

            foreach (var bodyPart in jdController.bodyPartsDict.Values)
            {
                bodyPart.Reset();
            }
        }

        isTrajectoryForked = forkToTrajectory;
        isNewDecisionStep = true;
        currentDecisionStep = 1;
    }
}
