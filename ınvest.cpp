#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;


class data_vis {int *entry_time,int *exit_time;}{


public:
    data_vis(int *entry_time,int *exit_time){
        this->entry_time=entry_time;
        this->exit_time=exit_time;
    }
    int get_entry_time(){
        return *entry_time;
    }
    int get_exit_time(){
        return *exit_time;
    }
    void set_entry_time(int entry_time){
        *entry_time=entry_time;
    }
    void set_exit_time(int exit_time){
        *exit_time=exit_time;
    }
}








