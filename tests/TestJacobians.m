classdef TestJacobians < TestSuperClass
    % The three As of unit testing: arrange, act, assert
    %
    % Summary of all tests performed:
    %
    % Copyright(C)2024, Institut fuer Mechatronische Systeme, Leibniz
    % Universitaet Hannover
    % Author: Max Bartholdt (max.bartholdt[at]imes.uni-hannover.de)
    %
    properties
        
    end
    
    methods (Test)
        %% Test with single joint
        function testRightJacobian(testCase)
            q = [1,1;];
            
            numJac = testCase.numJacobianSE3(@(t,q)testCase.kinematic(t,q),q);
            
            TT(:,:,1) = eye(4);
            for i = 1:size(q,1)
                TT(:,:,i+1) = SE3Group.expMap([0 0 1 q(i,:) 0]);
            end
            
            
            anaJac = testCase.geoJac(q,TT);
            
            anaJac = anaJac(:,[4:5]);
            err = testCase.equalJacobians(anaJac,numJac);
            testCase.verifyLessThan(err,testCase.tolerance_eigenvalue);
        end
        %% Test with double joint
        function testRightJacobianDoubleJoint(testCase)
            q = [1,1;2,3];
            
            numJac = testCase.numJacobianSE3(@(t,q)testCase.kinematic(t,reshape(q,[],2)),q(:));
            
            TT(:,:,1) = eye(4);
            for i = 1:size(q,1)
                TT(:,:,i+1) = SE3Group.expMap([0 0 1 q(i,:) 0]);
            end
            
            
            anaJac = testCase.geoJac(q,TT);
            
            anaJac = anaJac(:,[4,10,5,11]);
            err = testCase.equalJacobians(anaJac,numJac);
            testCase.verifyLessThan(err,testCase.tolerance_eigenvalue);
        end

        %% Test with triple joint
        function testRightJacobianTripleJoint(testCase)
            q = [1,1;2,3;-2,-1];
            
            numJac = testCase.numJacobianSE3(@(t,q)testCase.kinematic(t,reshape(q,[],2)),q(:));
            
            TT(:,:,1) = eye(4);
            for i = 1:size(q,1)
                TT(:,:,i+1) = SE3Group.expMap([0 0 1 q(i,:) 0]);
            end
            
            
            anaJac = testCase.geoJac(q,TT);
            
            anaJac = anaJac(:,[4,10,16,5,11,17]);
            err = testCase.equalJacobians(anaJac,numJac);
            testCase.verifyLessThan(err,testCase.tolerance_eigenvalue);
        end


        %% Test with triple joint
        function testLeftJacobianTripleJoint(testCase)
            q = [1,1;2,3;-2,-1];
            
            numJac = testCase.numJacobianSE3(@(t,q)testCase.kinematic(t,reshape(q,[],2)),q(:));
            
            TT(:,:,1) = eye(4);
            for i = 1:size(q,1)
                TT(:,:,i+1) = SE3Group.expMap([0 0 1 q(i,:) 0]);
            end
            
            
            anaJac = testCase.geoJac(q,TT);
            
            anaJac = anaJac(:,[4,10,16,5,11,17]);
            err = testCase.equalJacobians(anaJac,numJac);
            testCase.verifyLessThan(err,testCase.tolerance_eigenvalue);
        end
    end
    
    methods (Static)
        function out = kinematic(t,q)
            T = eye(4);
            for i = 1:size(q,1)
                T = T*SE3Group.expMap([0 0 1 q(i,:) 0]);
            end
            out = T;
            
        end
        function out = geoJac(q,TT)
            % Jacobian of a chained SE3 element T = TT(1)* TT(2)* TT(3)
            % w.r.t q1 q2 q3 with TT(2) = Exp(q1) ... ;
            out = [];
            for i = 1:size(q,1)
                Z = eye(4);
                % T = Y + q_i + Z = Q.Z
                % Z = I + q_i+1 + ... + q_n
                for j = i+2:size(TT,3)
                    Z = Z*TT(:,:,j);
                end
                
                % J_i = J(Q.Z)_Q * J(Q)_q = Ad(Z^-1) * J_r(q)
                J = SE3Group.rightJacobian([0 0 1 q(i,:) 0 ]);
                Ad_inv = SE3Group.adjoint(SE3Group.inverse(Z));
                out = [out, Ad_inv*J];
            end
        end
    end
end

