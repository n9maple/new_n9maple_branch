using UnityEngine;
using System.Collections.Generic;
using System;

public class VehicleController : MonoBehaviour
{
    public event Action OnHitWall;
    public event Action OnFinish;

    private const float MaxRaycastDistance = 10000;

    public float maxVelocity = 40f;
    public float turnFactor = 100;
    public Vector3 rayBias = new (0, 0.5f, 0);
    public LayerMask rayMask = ~0;
    public float guidingAngle = 0;
    public NN Fnn;
    private float _velocity;
    private Quaternion _rotation;
    private Collider _collider;

    public double[] inputs;

    private void Start()
    {
        _collider = GetComponent<Collider>();
    }
    
    private void FixedUpdate()
    {
        var turn = GetNnOutput();
        
        MoveVehicle(turn);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.name == "Ground")
            return;

        if (collision.collider.name == "Finishline")
            OnFinish?.Invoke();
        OnHitWall?.Invoke();
    }

    private float GetNnOutput()
    {
        inputs = new double[] { 
            GetDirectionDistance(new Vector3(-1, 0, 0)),
            GetDirectionDistance(new Vector3(-1, 0, 1)),
            GetDirectionDistance(new Vector3(0, 0, 1)),
            GetDirectionDistance(new Vector3(1, 0, 1)),
            GetDirectionDistance(new Vector3(1, 0, 0)),
        };
        
        //Debug.Log("ditances: " + string.Join(", ", distances));
        Fnn.SetInput(inputs);

        var outputs = Fnn.ForwardPass();
        var turn  = (float)outputs[0];
        //var friction = Mathf.Max(1f, frictionFactor * (float)outputs[2]);

        return turn;
    }

    private double GetDirectionDistance(Vector3 direction)
    {
        var worldDirection = transform.TransformDirection(direction.normalized);

        // Disable own collider, so raycast wouldn't hit it
        _collider.enabled = false;
        
        // Add ray bias to make sure ray doesn't hit the ground
        var rayPoint = transform.position + rayBias;
        var ray = new Ray(rayPoint, worldDirection);

        float distance = Physics.Raycast(ray, out var hit, MaxRaycastDistance, rayMask)
            ? hit.distance
            : MaxRaycastDistance;
           
        Debug.DrawRay(rayPoint, worldDirection * distance, Color.green);

        _collider.enabled = true;

        return distance;
    }

    private void MoveVehicle(float turn)
    {
        _velocity = maxVelocity;

        _rotation = transform.rotation;
        _rotation *= Quaternion.AngleAxis(-turn * turnFactor * Time.deltaTime, Vector3.up);
        
        var direction = _rotation * Vector3.forward;

        transform.position += direction * (_velocity * Time.deltaTime);
        transform.rotation = _rotation;
    }
}
