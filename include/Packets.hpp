#pragma once

#include "commands.pb.h"
#include <string>

namespace dtl
{
    namespace packets
    {
        constexpr const int ChildInfoRequest_Opcode = 0xA0;
        constexpr const int ChildInfoResponse_Opcode = 0xA1;
        constexpr const int ChildComplete_Opcode = 0xA2;
        constexpr const int ChildSetInfo_Opcode = 0xA3;
        constexpr const int IssueTaskRequest_Opcode = 0xA4;
        constexpr const int IssueTaskResponse_Opcode = 0xA5;
        constexpr const int TerminateAllChildrenRequest_Opcode = 0xA6;
        constexpr const int TerminateAllChildrenResponse_Opcode = 0xA7;

        struct SerializedPacket
        {
            char *data;
            size_t size;

            SerializedPacket()
            {
                this->data = nullptr;
                this->size = 0;
            }

            SerializedPacket(size_t size)
            {
                Initialize(size);
            }

            ~SerializedPacket()
            {
                Clear();
            }

            void Clear()
            {
                if (data) 
                    delete[] data;
                this->size = 0;
            }

            void Initialize(size_t s)
            {
                this->Clear();
                this->size = s;
                data = new char[s];
            }
        };

        template <typename T> bool SerializePacket(
            T& msg,
            SerializedPacket& pkt
        )
        {
            pkt.Initialize(msg.ByteSize());
            return msg.IsInitialized() && msg.SerializeToArray(pkt.data, pkt.size);
        }

        /**
         * Packet factory functions
         */

        ChildInfoRequest GetChildInfoRequestPacket()
        {
            ChildInfoRequest pkt;
            pkt.set_opcode(ChildInfoRequest_Opcode);
            return pkt;
        }

        ChildInfoResponse GetChildInfoResponsePacket(
            const std::string& name,
            int status,
            const std::string& currentFn,
            bool hasGPU
        )
        {
            ChildInfoResponse pkt;
            pkt.set_opcode(ChildInfoResponse_Opcode);
            pkt.set_name(name);
            pkt.set_status(status);
            pkt.set_currentfn(currentFn);
            pkt.set_hasgpu(hasGPU);
            return pkt;
        }

        ChildComplete GetChildCompletePacket(
            const std::string& name
        )
        {
            ChildComplete pkt;
            pkt.set_opcode(ChildComplete_Opcode);
            pkt.set_name(name);
            return pkt;
        }

        ChildSetInfo GetChildSetInfoPacket(
            const std::string& name
        )
        {
            ChildSetInfo pkt;
            pkt.set_opcode(ChildSetInfo_Opcode);
            pkt.set_name(name);
            return pkt;
        }

        IssueTaskRequest GetIssueTaskRequest(
            const std::string& name,
            const std::string& function,
            bool needsGPU,
            bool hasParameters
        )
        {
            IssueTaskRequest pkt;
            pkt.set_opcode(IssueTaskRequest_Opcode);
            pkt.set_name(name);
            pkt.set_function(function);
            pkt.set_needsgpu(needsGPU);
            pkt.set_hasparameters(hasParameters);
            return pkt;
        }

        IssueTaskResponse GetIssueTaskResponse(
            const std::string& name,
            bool accepted
        )
        {
            IssueTaskResponse pkt;
            pkt.set_opcode(IssueTaskResponse_Opcode);
            pkt.set_name(name);
            pkt.set_accepted(accepted);
            return pkt;
        }

        TerminateAllChildrenRequest GetTerminateAllChildrenRequestPacket()
        {
            TerminateAllChildrenRequest pkt;
            pkt.set_opcode(TerminateAllChildrenRequest_Opcode);
            return pkt;
        }

        TerminateAllChildrenResponse GetTerminateAllChildrenResponsePacket(
            const std::string& name
        )
        {
            TerminateAllChildrenResponse pkt;
            pkt.set_opcode(TerminateAllChildrenResponse_Opcode);
            pkt.set_name(name);
            return pkt;
        }

        /**
         * Packet parsing functions
         */

        template<typename T> bool ParseDtlPacket(
            void *buffer,
            int len,
            int opcode,
            T& pkt
        )
        {
            bool ok = pkt.ParseFromArray(buffer, len);
            return ok && (pkt.opcode() == opcode);
        }

        bool ParseChildInfoRequestPacket(
            void *buffer,
            int len,
            ChildInfoRequest& pkt
        )
        {
            return ParseDtlPacket(buffer, len, ChildInfoRequest_Opcode, pkt);
        }

        bool ParseChildInfoResponsePacket(
            void *buffer,
            int len,
            ChildInfoResponse& pkt
        )
        {
            return ParseDtlPacket(buffer, len, ChildInfoResponse_Opcode, pkt);
        }
        
        bool ParseChildCompletePacket(
            void *buffer,
            int len,
            ChildComplete& pkt
        )
        {
            return ParseDtlPacket(buffer, len, ChildComplete_Opcode, pkt);
        }

        bool ParseChildSetInfoPacket(
            void *buffer,
            int len,
            ChildSetInfo& pkt
        )
        {
            return ParseDtlPacket(buffer, len, ChildSetInfo_Opcode, pkt);
        }

        bool ParseIssueTaskRequestPacket(
            void *buffer,
            int len,
            IssueTaskRequest& pkt
        )
        {
            return ParseDtlPacket(buffer, len, IssueTaskRequest_Opcode, pkt);
        }

        bool ParseIssueTaskResponsePacket(
            void *buffer,
            int len,
            IssueTaskResponse& pkt
        )
        {
            return ParseDtlPacket(buffer, len, IssueTaskResponse_Opcode, pkt);
        }

        bool ParseTerminateAllChildrenRequestPacket(
            void *buffer,
            int len,
            TerminateAllChildrenRequest& pkt
        )
        {
            return ParseDtlPacket(buffer, len, TerminateAllChildrenRequest_Opcode, pkt);
        }

        bool ParseTerminateAllChildrenResponsePacket(
            void *buffer,
            int len,
            TerminateAllChildrenResponse& pkt
        )
        {
            return ParseDtlPacket(buffer, len, TerminateAllChildrenResponse_Opcode, pkt);
        }
    }
}
