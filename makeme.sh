#!/bin/bash

# Exit the script if an error occurs
set -e

function check_submodule ()
{
    submodule_name=${1}
    test_file=${2}
    main_branch=${3}
    if [ -d "${submodule_name}/${test_file}" ]; then
        echo "${submodule_name} exists"
    else
        echo "${submodule_name} does not exist"
    	echo "git submodules will be updated"
        rm -r ${submodule_name}
        git submodule update --init
        pushd .
        cd ${submodule_name}
        git pull origin ${main_branch}
        popd
    fi
}

function do_the_build () {
    echo ""

    if [ $( git rev-parse --is-inside-work-tree ) ]; then
        check_submodule "viz/glad" "src" "main"
        check_submodule "viz/glfw" "src" "master"
        check_submodule "viz/glm" "glm" "master"
        check_submodule "viz/imgui" "backends" "master"
        check_submodule "tests/exec_tests/Catch2_header_only" "catch2.hpp" "main"
    else
        echo "this is not a repository"
    fi

    pushd .
    popd
    pushd .
    cd build

    # Either pass the explicit help arg to the cmake file or give all function
    # args to the cmake file
    if [ "${1}" == "--help" ]; then
        cmake -DHELP_SOS=1 -LAH ..
    else
        cmake $@ ..
        num_jays=1
        if [ -z $( which nproc ) ]; then
            echo "the nproc command wasn't found, using two cores to build."
            num_jays=4
        else
            if [ $( nproc ) -ge 4 ]; then
                num_jays=4
            else
                echo "Using one core to build since there are only " $( nproc ) " available"
            fi
        fi
        make -j ${num_jays}
    fi

    popd
}

if [ "${1}" == "--help" ]; then
    args=$1
else
    args=$@
fi

if [ -d build ]; then
    do_the_build $args
else
    mkdir build
    do_the_build $args
fi
