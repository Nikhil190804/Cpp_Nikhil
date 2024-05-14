#include<iostream>
using namespace std;

void Selection_Sort(int arr[],int size){
    for(int i=0;i<size;i++){
        int minimum_index=i;
        for(int j=i+1;j<size;j++){
            if(arr[j]<arr[minimum_index]){
                minimum_index=j;
                
            }
            swap(arr[i],arr[minimum_index]);
        }
    }

}
int main(){
    int n=9;
    int array[]={2,1,4,6,3,7,1,9,6};
    Selection_Sort(array,n);
    for(auto x:array){
        cout<<x<<endl;
    }
return 0;
}