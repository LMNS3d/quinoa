package Quinoa_2_Mac.buildTypes

import jetbrains.buildServer.configs.kotlin.v10.*
import jetbrains.buildServer.configs.kotlin.v10.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.v10.buildSteps.ScriptBuildStep.*
import jetbrains.buildServer.configs.kotlin.v10.buildSteps.script

object Quinoa_2_Mac_Matrix : Template({
    uuid = "75d7a7e0-9c7c-4bc4-9cbb-3d1d8e58febd"
    extId = "Quinoa_2_Mac_Matrix"
    name = "Matrix"

    vcs {
        root(Quinoa_2.vcsRoots.Quinoa_2_GitGithubComQuinoacomputingQuinoaGitRefsHeadsMaster)

    }

    steps {
        script {
            name = "Verify commit"
            id = "RUNNER_20"
            scriptContent = """git verify-commit %build.vcs.number% 2>&1 | grep "Good signature""""
        }
        script {
            name = "Build code"
            id = "RUNNER_21"
            scriptContent = """
                rm -rf build && mkdir build && cd build
                cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DCMAKE_BUILD_TYPE=%buildtype% -DCMAKE_CXX_FLAGS=-Werror -DTPL_DIR=/Volumes/Storage/jbakosi/code/quinoa-tpl/install/%compiler%-x86_64 ../src
                make -j%teamcity.agent.hardware.cpuCount%
            """.trimIndent()
        }
        script {
            name = "Run tests"
            id = "RUNNER_22"
            workingDir = "build"
            scriptContent = "../script/run_tests.sh"
        }
    }

    requirements {
        equals("teamcity.agent.jvm.os.name", "Mac OS X", "RQ_21")
        contains("teamcity.agent.name", "euler", "RQ_22")
    }
})