#include<iostream>
using namespace std;

void Bubble_Sort(int arr[],int size){
    bool swapped=false;         //for optimising solution in case of sorted array
    for(int i=0;i<size-1;i++){
        for(int j=0;j<size-i-1;j++){
            if(arr[j]<=arr[j+1]){
                ;
            }
            else{
                swap(arr[j],arr[j+1]);
                swapped=true;
            }
        }
        if(swapped==false){
            break;      //already sorted
        }
    }
}


int main(){
    int n=9;
    int array[]={2,1,4,6,3,7,1,9,6};
    Bubble_Sort(array,n);
    for(auto x:array){
        cout<<x<<endl;
    }
return 0;
}