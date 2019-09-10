using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class HighLevelPlanner : Agent
{
    public ClimberAgent ActingAgent;

    public override void InitializeAgent()
    {
        agentParameters.onDemandDecision = true;
        agentParameters.resetOnDone = true;
        agentParameters.numberOfActionsBetweenDecisions = 5;
        agentParameters.maxStep = 0;
    }

    public bool IsAgentInfoSent()
    {
        return GetReward() == 0;// begining of an action
    }

    public void AddHighLevelObservation(float observation)
    {
        AddVectorObs(observation);
    }

    public void AddHighLevelObservation(int observation)
    {
        AddVectorObs(observation);
    }

    public void AddHighLevelObservation(Vector3 observation)
    {
        AddVectorObs(observation);
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations()
    {
        //ActingAgent.GetObservationForHighLevelPlanner();
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        //ActingAgent.SetTaskFromHighLevelplanner(vectorAction);
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void AgentReset()
    {
        
    }
}
